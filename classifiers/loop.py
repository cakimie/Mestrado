import pandas as pd
import numpy as np

from clearml import Task

from clearml import PipelineController

def create_tasks(K, country=None, city=None, category=None):
    filter_str = ''
    if country is not None:
        filter_str = filter_str + f'_{country}'
    if city is not None:
        filter_str = filter_str + f'_{city}'
    if category is not None:
        filter_str = filter_str + f'_{category}'
    classifier_nodes = []
    for queue, classifier in classifiers:
        kfold_nodes = []
        for k in range(K):
            node_name = f'{classifier}{filter_str}_f{k+1}-{K}'
            kfold_nodes.append(node_name)
            create_task(node_name, queue, classifier, k+1, K, country, city, category)
        kfold_results_name = f'{K}-fold_results_{classifier}{filter_str}'
        classifier_nodes.append(kfold_results_name)
        pipe.add_function_step(
            name=kfold_results_name,
            execution_queue='default',
            parents=kfold_nodes,
            function=kfold_results,
            function_kwargs={
                **{ 'date_time': date_time },
                **{ node_name: '${%s.id}' % node_name for node_name in kfold_nodes },
            },
        )
    pipe.add_function_step(
        name=f'classifiers{filter_str}_result',
        execution_queue='default',
        parents=classifier_nodes,
        function=classifier_results,
        function_kwargs={
            **{ 'date_time': date_time },
            **{ node_name: '${%s.id}' % node_name for node_name in classifier_nodes },
        },
    )

def create_task(new_name, queue, classifier, k, K, country=None, city=None, category=None):
    print(f'Creating task for fold {k}/{K} of {classifier}')
    # task = Task.get_task(project_name='PopularTimesFold/Classifier', task_name=classifier)
    # reported_metrics = task.get_reported_scalars()['metrics'].keys()
    # print(f'...will monitor metrics {reported_metrics}')
    pipe.add_step(
        name=new_name,
        execution_queue=queue,
        base_task_project='PopularTimesFold/Classifier',
        base_task_name=classifier,
        # monitor_metrics=[('metrics', metric) for metric in reported_metrics],
        parameter_override={
            'General/k': k,
            'General/K': K,
            'General/country': country,
            'General/city': city,
            'General/category': category,
        }
    )

# -----------------------------------------------------------------------------------
# NOTE: the following functions are "teleported" to another script before execution;
# thus, they DO NOT share the same global scope as the functions above, and therefore
# MUST receive the relevant variables as parameters (kwargs):

def kfold_results(**kwargs):
    date_time=kwargs['date_time']
    del kwargs['date_time']

    # For each incoming node, compare against current best
    metrics_acum = {}
    i = 0
    for node_name, fold_task_id in kwargs.items():
        i += 1
        print(i, node_name, fold_task_id)
        # Get the original task based on the ID we got from the pipeline
        task = Task.get_task(task_id=fold_task_id)
        task.add_tags([date_time])
        # accuracy = task.get_reported_scalars()['metrics']['accuracy']['y'][0]
        reported_metrics = task.get_reported_scalars()['metrics']
        for metric in reported_metrics.keys():
            metric_value = reported_metrics[metric]['y'][0]
            print(f'Found in {node_name} {metric}={metric_value}')
            if metric in metrics_acum:
                metrics_acum[metric].append(metric_value)
            else:
                metrics_acum[metric] = [metric_value]
            Task.current_task().get_logger().report_scalar('metrics', metric, iteration=i, value=metric_value)
    for metric, values in metrics_acum.items():
        Task.current_task().get_logger().report_scalar('metrics', metric, iteration=len(values)+1, value=np.mean(values))

    # Print the final best model details and log it as an output model on this step
    print(f"Results for {Task.current_task().name}: {metrics_acum}")

def classifier_results(**kwargs):
    date_time=kwargs['date_time']
    del kwargs['date_time']

    # For each incoming node, compare against current best
    metrics_acum = {}
    i = 0
    for node_name, classifier_task_id in kwargs.items():
        i += 1
        print(i, node_name, classifier_task_id)
        # Get the original task based on the ID we got from the pipeline
        task = Task.get_task(task_id=classifier_task_id)
        task.add_tags([date_time])
        reported_metrics = task.get_reported_scalars()['metrics']
        for metric in reported_metrics.keys():
            metric_values = reported_metrics[metric]['y']
            print(f'Found in {node_name} {metric}={metric_values}')
            if metric in metrics_acum:
                metrics_acum[metric].append(metric_values)
            else:
                metrics_acum[metric] = [metric_values]
            Task.current_task().get_logger().report_scalar('metrics', f'{metric}', iteration=i, value=metric_values[-1])

    # Print the final best model details and log it as an output model on this step
    print(f"Results for {Task.current_task().name}: {metrics_acum}")

# ------------------------------ START ------------------------------------

# Queue name and task name for every classifier taking part in the pipeline:
classifiers = [
    ['light', 'DrCIF'],
    ['heavy', 'elastic_ensemble'],
    ['light', 'hydra_ridge'],
    ['default', 'MrSQM'],
    ['heavy', 'multirocket'],
    ['heavy', 'proximity_forest'],
    ['light', 'rdst'],
    ['light', 'ridge_cv'],
    ['default', 'rSTSF'],
]

# Extracts types of possible filters to pick from later:
df = pd.read_csv('weekdays_datasets/df_timeseries.csv')
unique_categories = df[['country', 'city', 'category']].drop_duplicates()
unique_cities = unique_categories[['country', 'city']].drop_duplicates()
unique_countries = unique_cities[['country']].drop_duplicates()
unique_categories = np.array(unique_categories, dtype=int).tolist()       # 2D
unique_cities = np.array(unique_cities, dtype=int).tolist()               # 2D
unique_countries = np.array(unique_countries, dtype=int).ravel().tolist() # 1D

K = 5

# Creates pipeline controller:
pipe = PipelineController(
    project='PopularTimesFold/Classifier/Pipelines',  # NOTE: pipeline tasks will be created in this project.
    name=f'City by city {K}-fold', # NOTE: edit this if changing filter mode below.
    version="1.0"
)

from datetime import datetime
now = datetime.now() # current date and time
date_time = now.strftime('%Y-%d-%m_%H-%M-%S')

# NOTE: Pick one out of the filter modes below to create the pipeline architecture:
# for country, city, category in unique_categories: # This one doesn't make sense!
#     create_tasks(K, country, city, category)
for country, city in unique_cities:               # This one trains models city by city.
    create_tasks(K, country, city)
# for country in unique_countries:                    # This one trains models country by country.
#     create_tasks(K, country)
# create_tasks(K)                                   # This one trains models with full dataset.

# Initializes task queue from pipeline architecture:
pipe.start()

print('Done!')