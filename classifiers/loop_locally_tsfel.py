import pandas as pd
import numpy as np
import scipy.stats as stats
from clearml import Task
import torch

# Verificação de disponibilidade da GPU
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(10, 10).to(device)

# Função para criar tarefas
def create_tasks(K, target=None):
    filter_str = ''
    if target is not None:
        filter_str = f'_{target}'
    
    for queue, classifier in classifiers:
        kfold_results = []
        for k in range(K):
            node_name = f'{classifier}{filter_str}_f{k+1}-{K}'
            results = classifier(task_name=node_name,
                                 clearML=False,
                                params={'k': k + 1, 
                                        'K': K,
                                        'country': None,
                                        'city': None,
                                        'category': None,
                                },
                                dataset_filename = '/Users/carolina/Desktop/Mestrado/df_timeseries_with_tsfel_features_clean.csv',
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

# ------------------------------ START ------------------------------------
# Importação dos classificadores
# from classifiers.weasel_d import run_weasel_d
# from classifiers.tde import run_tde
from classifiers.rdst import run_rdst
# from classifiers.hydra_ridge import run_hydra_ridge

classifiers = [
    # ['heavy', run_weasel_d],
    # ['light', run_hydra_ridge],
    # ['light', run_tde],
    ['light', run_rdst],
]

K = 5

from datetime import datetime
now = datetime.now()
date_time = now.strftime('%Y-%d-%m_%H-%M-%S')

# Ajuste na lógica de criação de tarefas para usar a nova coluna target
filters_means = []

metrics = ['accuracy_score', 'f1_score', 'precision_score', 'recall_score']

for m, metric in enumerate(metrics):
    mean_of_means = np.mean([filter_means[m] for filter_means in filters_means])
    print(f'Classifier mean {metric}={mean_of_means:.7f}')

# Executa a função para todo o dataset
create_tasks(K)  # Esta linha treina modelos com o dataset completo.

print('Done!')
