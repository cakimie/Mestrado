import pandas as pd
import numpy as np
import scipy.stats as stats

from clearml import Task

import torch

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

x = torch.randn(10, 10).to(device)

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
        kfold_results = []
        for k in range(K):
            node_name = f'{classifier}{filter_str}_f{k+1}-{K}'
            # task=Task.create(project_name='PopularTimesFold/Classifier', task_name=node_name)
            results = classifier(task_name=node_name,
                                 clearML=False,
                                params={'k':k+1, 
                                        'K':K,
                                        'country':country, 
                                        'city':city,
                                        'category':category,
                                },
            )
            print(f'{node_name}', results)
            kfold_results.append(results)

        #TODO: Totalizar algoritmo com este filtro para todos os diferentes folds.
        metrics =  ['accuracy_score','f1_score','precision_score','recall_score']
        means = []

        print(f'{classifier}{filter_str}')
        for metric in metrics:
            values = [
                results[metric] for results in kfold_results
            ]
            metric_mean = np.mean(values)
            metric_std = np.std(values)
            trust_interval = stats.t.interval(0.95, K - 1, loc=metric_mean, scale=metric_std)
            means.append(metric_mean)
            print('          ',metric,f'mean={metric_mean:.7f}',f'std={metric_std:.7f}',f't_interval={trust_interval}')
        return means

# ------------------------------ START ------------------------------------
# ALEXANDRE: Aqui você descomenta os classificadores que for rodar
#            rode 1 por vez
#            caso o classificador não esteja na lista, basta incluí-lo conforme os imports abaixo

# from classifiers.fresh_prince import run_fresh_prince
# from classifiers.ts_fresh import run_ts_fresh
# from classifiers.hivecotev2 import run_hivecotev2
# from classifiers.weasel_d import run_weasel_d
# from classifiers.resnet import run_resnet
# from classifiers.tde import run_tde
# from classifiers.inception_time import run_inception_time
# from classifiers.drcif2 import run_DrCIF
# from classifiers.multirocket import run_multirocket
# from classifiers.rdst import run_rdst
# from classifiers.ridgecv import run_ridge_cv
# from classifiers.rSTSF import run_rSTSF
# from classifiers.elastic_ensemble import run_elastic_ensemble
# from classifiers.proximity_forest import run_proximity_forest
# from classifiers.hydra_ridge import run_hydra_ridge

#treino com um país e teste com outro (vice-versa)
# from classifiers.weasel_d_inverted import run_weasel_d
from classifiers.tde_inverted import run_tde
# from classifiers.hydra_ridge_inverted import run_hydra_ridge
# from classifiers.rdst_inverted import run_rdst
# from classifiers.rSTSF_inverted import run_rSTSF

#teste target
# from classifiers.weasel_d_multi_test import run_weasel_d
# from classifiers.rdst_multioutput import run_rdst
# from classifiers.rSTSF_multioutput import run_rSTSF
# from classifiers.hydra_ridge_multioutput import run_hydra_ridge
# from classifiers.weasel_d_target import run_weasel_d

# Queue name and task name for every classifier taking part in the pipeline:
#ALEXANDRE: Aqui você também deve descomentar os classificadores que for rodar
#           rodar 1 por vez
#           caso o classificador não esteja na lista, basta incluí-lo conforme os modelos abaixo
#           caso não saiba se é "light", "default" ou "heavy", pode deixá-lo como "default"

classifiers = [
    # ['light', run_fresh_prince], ok
    # ['light', run_ts_fresh], #ok
    # ['heavy', run_weasel_d], #ok
    ['heavy', run_tde], #ok
    # ['heavy', run_resnet],
    # ['heavy', run_hivecotev2], ok
    # ['heavy', run_inception_time],
    # ['light', run_DrCIF],
    # ['light', run_multirocket],
    # ['light', run_rdst],
    # ['light', run_ridge_cv],
    # ['light', run_rSTSF],
    # ['light', run_MrSQM],
    # ['heavy', run_elastic_ensemble], ok
    # ['heavy', run_proximity_forest],
    # ['heavy', run_hydra_ridge],
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

from datetime import datetime
date_time_init = datetime.now() # current date and time
start_time = date_time_init.strftime('%Y-%d-%m_%H-%M-%S')
print('Start time:', start_time)

# NOTE: Pick one out of the filter modes below to create the pipeline architecture:
# for country, city, category in unique_categories: # This one doesn't make sense!
#     create_tasks(K, country, city, category)
filters_means = []

# for country, city in unique_cities:               # This one trains models city by city.
#     means = create_tasks(K, country, city)
#     filters_means.append(means)

# for country in unique_countries:               # This one trains models country by country.
#     means = create_tasks(K, country)
#     filters_means.append(means)

metrics =  ['accuracy_score','f1_score','precision_score','recall_score']

for m,metric in enumerate(metrics):
    mean_of_means = np.mean([filter_means[m] for filter_means in filters_means])
    print(f'Classifier mean {metric}={mean_of_means:.7f}')

    #TODO: Totalizar algoritmo para todos os diferentes filtros.

for country in unique_countries:                    # This one trains models country by country.
    create_tasks(K, country)

# create_tasks(K)                                   # This one trains models with full dataset.

date_time_finish = datetime.now() # current date and time
end_time = date_time_finish.strftime('%Y-%d-%m_%H-%M-%S')
print('End time:', end_time)

running_time = date_time_finish - date_time_init
print('Running time:', str(running_time))

print('Done!')