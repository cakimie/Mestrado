from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning
from aeon.classification.shapelet_based import RDSTClassifier
import warnings

# Função para calcular métricas de forma segura
def rdst(X_train, y_train, X_test, y_test):
    clf_rdst = RDSTClassifier()
    clf_rdst.fit(X_train, y_train)
    rdst_pred = clf_rdst.predict(X_test)
    
    metrics = {}

    # Ignora warnings de métricas indefinidas (por exemplo, se uma classe não for prevista)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        # Calcula as métricas e captura as exceções de classes ausentes
        try:
            metrics['accuracy_score'] = accuracy_score(y_test, rdst_pred)
        except Exception as e:
            metrics['accuracy_score'] = f"Erro: {str(e)}"

        try:
            metrics['f1_score'] = f1_score(y_test, rdst_pred, average='weighted')
        except Exception as e:
            metrics['f1_score'] = f"Erro: {str(e)}"

        try:
            metrics['precision_score'] = precision_score(y_test, rdst_pred, average='weighted')
        except Exception as e:
            metrics['precision_score'] = f"Erro: {str(e)}"

        try:
            metrics['recall_score'] = recall_score(y_test, rdst_pred, average='weighted')
        except Exception as e:
            metrics['recall_score'] = f"Erro: {str(e)}"
    
    return metrics

# Função para rodar a classificação RDST
def run_rdst(
    clearML=False,
    params={
        'k': 1,
        'K': 10,
        'country': 0,
        'city': 0,
        'category': None,
    },
    task=None,
    task_name="rdst",
):
    import time
    import numpy as np
    import pandas as pd
    from classifiers.load_fold import load_fold

    start_time = time.time()

    if clearML:
        from clearml import Task
        if task is None:
            task = Task.init(project_name='PopularTimesFold/Classifier', task_name="rdst")
        task.connect(params)

    # Carrega o dataset
    df = pd.read_csv('weekdays_datasets/df_timeseries.csv')

    # Separa os dados por país
    df_country_0 = df[df['country'] == 0] 
    df_country_1 = df[df['country'] == 1] 

    # Dados de treino: país 0, dados de teste: país 1
    X_train = df_country_0.drop(columns=['category'])
    y_train = df_country_0['category']
    X_test = df_country_1.drop(columns=['category'])
    y_test = df_country_1['category']

    print(f'Treinando no país 0 e testando no país 1')
    results = rdst(X_train, y_train, X_test, y_test)

    # Dados de treino: país 1, dados de teste: país 0
    print(f'Treinando no país 1 e testando no país 0')
    X_train = df_country_1.drop(columns=['category'])
    y_train = df_country_1['category']
    X_test = df_country_0.drop(columns=['category'])
    y_test = df_country_0['category']

    results_inverted = rdst(X_train, y_train, X_test, y_test)

    # Se estiver usando ClearML, registre os resultados
    if clearML:
        task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - start_time)
        for key, value in results.items():
            task.get_logger().report_scalar('metrics', f'{key}_train_country_0_test_country_1', iteration=0, value=value)
        for key, value in results_inverted.items():
            task.get_logger().report_scalar('metrics', f'{key}_train_country_1_test_country_0', iteration=0, value=value)
        task.close()

    return results, results_inverted

if __name__ == '__main__':
    results, results_inverted = run_rdst()
    print(results)
    print(results_inverted)
