from clearml import Task
import numpy as np
import scipy.stats as stats


results_list = [
    '5-fold_results_rSTSF_0_0',
    '5-fold_results_rSTSF_0_1',
    '5-fold_results_rSTSF_0_2',
    '5-fold_results_rSTSF_1_0',
    '5-fold_results_rSTSF_1_2',
    '5-fold_results_rSTSF_1_4',
    '5-fold_results_DrCIF_0_0',
    '5-fold_results_DrCIF_0_1',
    '5-fold_results_DrCIF_0_2',
    '5-fold_results_DrCIF_1_0',
    '5-fold_results_DrCIF_1_2',
    '5-fold_results_DrCIF_1_4',
    '5-fold_results_elastic_ensemble_0_0',
    '5-fold_results_elastic_ensemble_0_1',
    '5-fold_results_elastic_ensemble_0_2',
    '5-fold_results_elastic_ensemble_1_0',
    '5-fold_results_elastic_ensemble_1_2',
    '5-fold_results_elastic_ensemble_1_4',
    '5-fold_results_hydra_ridge_0_0',
    '5-fold_results_hydra_ridge_0_1',
    '5-fold_results_hydra_ridge_0_2',
    '5-fold_results_hydra_ridge_1_0',
    '5-fold_results_hydra_ridge_1_2',
    '5-fold_results_hydra_ridge_1_4',
    '5-fold_results_MrSQM_0_0',
    '5-fold_results_MrSQM_0_1',
    '5-fold_results_MrSQM_0_2',
    '5-fold_results_MrSQM_1_0',
    '5-fold_results_MrSQM_1_2',
    '5-fold_results_MrSQM_1_4',
    '5-fold_results_multirocket_0_0',
    '5-fold_results_multirocket_0_1',
    '5-fold_results_multirocket_0_2',
    '5-fold_results_multirocket_1_0',
    '5-fold_results_multirocket_1_2',
    '5-fold_results_multirocket_1_4',
    '5-fold_results_proximity_forest_0_0',
    '5-fold_results_proximity_forest_0_1',
    '5-fold_results_proximity_forest_0_2',
    '5-fold_results_proximity_forest_1_0',
    '5-fold_results_proximity_forest_1_2',
    '5-fold_results_proximity_forest_1_4',
    '5-fold_results_rdst_0_0',
    '5-fold_results_rdst_0_1',
    '5-fold_results_rdst_0_2',
    '5-fold_results_rdst_1_0',
    '5-fold_results_rdst_1_2',
    '5-fold_results_rdst_1_4',
    '5-fold_results_ridge_cv_0_0',
    '5-fold_results_ridge_cv_0_1',
    '5-fold_results_ridge_cv_0_2',
    '5-fold_results_ridge_cv_1_0',
    '5-fold_results_ridge_cv_1_2',
    '5-fold_results_ridge_cv_1_4',
]

for result in results_list:
    task = Task.get_task(task_name=result)
    if task is None:
        print('!!!!! ------- ERRO!!! ------- !!!!!')
        print(f'Não foi possível localizar resultados de {result}')
        continue
    print(f'Results for {result} {task.get_status()}')

    metrics = task.get_reported_scalars()['metrics'].keys()

    # Dados prontos para Ctrl+C:
    for m in metrics:
        data = task.get_reported_scalars()['metrics'][m]
        values = data['y'][1:]
        print(f'{np.mean(values):.10f}')

    # Dados para debugar:
    for m in metrics:
        data = task.get_reported_scalars()['metrics'][m]
        values = data['y'][1:]
        print(m, f'mean={np.mean(values):.10f}', f'std={np.std(values):.10f}',f'trust_interval={stats.t.interval(0.95, 5 - 1, loc=np.mean(values), scale=np.std(values))}')

    if task.get_status() != 'published':
        task.publish(True)