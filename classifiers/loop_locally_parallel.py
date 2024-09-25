import pandas as pd
import numpy as np
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor
from clearml import Task

def run_classifier_in_parallel(queue, classifier, filter_str, K, country=None, city=None, category=None):
    kfold_results = []
    for k in range(K):
        node_name = f'{classifier}{filter_str}_f{k+1}-{K}'
        # task = Task.create(project_name='PopularTimesFold/Classifier', task_name=node_name)
        results = classifier(task_name=node_name,
                             clearML=False,
                             params={'k': k+1,
                                     'K': K,
                                     'country': country,
                                     'city': city,
                                     'category': category,
                             },
        )
        print(f'{node_name}', results)
        kfold_results.append(results)

    metrics = ['accuracy_score', 'f1_score', 'precision_score', 'recall_score']
    means = []

    print(f'{classifier}{filter_str}')
    for metric in metrics:
        values = [results[metric] for results in kfold_results]
        metric_mean = np.mean(values)
        metric_std = np.std(values)
        trust_interval = stats.t.interval(0.95, K - 1, loc=metric_mean, scale=metric_std)
        means.append(metric_mean)
        print('          ', metric, f'mean={metric_mean:.7f}', f'std={metric_std:.7f}', f't_interval={trust_interval}')
    
    return means

def create_tasks(K, country=None, city=None, category=None):
    filter_str = ''.join([f'_{country}' if country else '', 
                         f'_{city}' if city else '', 
                         f'_{category}' if category else ''])

    classifier_nodes = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for queue, classifier in classifiers:
            futures.append(executor.submit(run_classifier_in_parallel, queue, classifier, filter_str, K, country, city, category))

        # Espera até que todas as tarefas paralelas terminem e coleta os resultados
        for future in futures:
            result = future.result()
            classifier_nodes.append(result)

    return classifier_nodes

# ------------------------------ START ------------------------------------

from classifiers.weasel_d import run_weasel_d

classifiers = [
    ['light', run_weasel_d],
]

df = pd.read_csv('weekdays_datasets/df_timeseries.csv')
unique_categories = df[['country', 'city', 'category']].drop_duplicates()
unique_cities = unique_categories[['country', 'city']].drop_duplicates()
unique_countries = unique_cities[['country']].drop_duplicates()
unique_categories = np.array(unique_categories, dtype=int).tolist()       # 2D
unique_cities = np.array(unique_cities, dtype=int).tolist()               # 2D
unique_countries = np.array(unique_countries, dtype=int).ravel().tolist() # 1D

K = 5

from datetime import datetime
now = datetime.now()
date_time = now.strftime('%Y-%d-%m_%H-%M-%S')

filters_means = []

# Escolha o filtro desejado:
# for country, city in unique_cities:  
#     means = create_tasks(K, country, city)
#     filters_means.append(means)

# for country in unique_countries:  
#     means = create_tasks(K, country)
#     filters_means.append(means)

metrics = ['accuracy_score', 'f1_score', 'precision_score', 'recall_score']

for m, metric in enumerate(metrics):
    mean_of_means = np.mean([filter_means[m] for filter_means in filters_means])
    print(f'Classifier mean {metric}={mean_of_means:.7f}')

create_tasks(K)
print('Done!')
