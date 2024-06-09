from clearml import Task
import numpy as np
import scipy.stats as stats


results_list = [
    # 'multirocket_1_0_f1-5',
    # 'multirocket_1_0_f2-5',
    # 'multirocket_1_0_f3-5',
    # 'multirocket_1_0_f4-5',
    # 'multirocket_1_0_f5-5',
    'multirocket_1_2_f1-5',
    'multirocket_1_2_f2-5',
    'multirocket_1_2_f3-5',
    'multirocket_1_2_f4-5',
    'multirocket_1_2_f5-5',
]

values_arr={
        'accuracy_score':[],
        'f1_score':[],
        'precision_score':[],
        'recall_score':[],
    }

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
        values = data['y'][0]
        values_arr[m].append(values)
        print(f'  {m}={values}')
        # print(f'valores={values}')

    if task.get_status() != 'published':
        task.publish(True)

print('Totalizando...', values_arr)

# Dados para debugar:
for m in metrics:
    # data = task.get_reported_scalars()['metrics'][m]
    # values = data['y'][0]
    print(m, f'mean={np.mean(values_arr[m]):.10f}', f'std={np.std(values_arr[m]):.10f}',f'trust_interval={stats.t.interval(0.95, 5 - 1, loc=np.mean(values_arr[m]), scale=np.std(values_arr[m]))}')
