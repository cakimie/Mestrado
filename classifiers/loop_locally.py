import pandas as pd
import numpy as np
import scipy.stats as stats

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
            kfold_results.append(classifier(task_name=node_name, 
                                            params={'k':k+1, 
                                                    'K':K,
                                                    'country':country, 
                                                    'city':city,
                                                    'category':category,
                                            }
            ))

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

from classifiers.fresh_prince import run_fresh_prince

# Queue name and task name for every classifier taking part in the pipeline:
classifiers = [
    ['light', run_fresh_prince],
    # ['light', run_ts_fresh],
    # ['light', run_weasel_d],
    # ['default', run_tde],
    # ['heavy', run_resnet],
    # ['heavy', run_hivecotev2],
    # ['heavy', run_inception_time],
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
now = datetime.now() # current date and time
date_time = now.strftime('%Y-%d-%m_%H-%M-%S')

# NOTE: Pick one out of the filter modes below to create the pipeline architecture:
# for country, city, category in unique_categories: # This one doesn't make sense!
#     create_tasks(K, country, city, category)
filters_means = []
for country, city in unique_cities:               # This one trains models city by city.
    means = create_tasks(K, country, city)
    filters_means.append(means)

metrics =  ['accuracy_score','f1_score','precision_score','recall_score']

for m,metric in enumerate(metrics):
    mean_of_means = np.mean([filter_means[m] for filter_means in filters_means])
    print(f'Classifier mean {metric}={mean_of_means:.7f}')

    #TODO: Totalizar algoritmo para todos os diferentes filtros.

# for country in unique_countries:                    # This one trains models country by country.
#     create_tasks(K, country)
# create_tasks(K)                                   # This one trains models with full dataset.

print('Done!')