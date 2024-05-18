from clearml import Task
import numpy as np


results_list = [
    '5-fold_results_rSTSF_0_0',
    '5-fold_results_rSTSF_0_1',
    '5-fold_results_rSTSF_0_2',
    '5-fold_results_rSTSF_1_0',
    '5-fold_results_rSTSF_1_2',
    '5-fold_results_rSTSF_1_4',
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
    print(f'Results for {result} {task.get_status()}')

    metrics = task.get_reported_scalars()['metrics'].keys()

    # Dados prontos para Ctrl+C:
    for m in metrics:
        data = task.get_reported_scalars()['metrics'][m]
        values = data['y'][1:]
        print(f'{np.mean(values):.7f}')

    # Dados para debugar:
    for m in metrics:
        data = task.get_reported_scalars()['metrics'][m]
        values = data['y'][1:]
        print(m, f'mean={np.mean(values):.7f}', f'std={np.std(values):.7f}')

    if task.get_status() != 'published':
        task.publish(True)