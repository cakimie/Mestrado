from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from weasel.classification.dictionary_based import WEASEL_V2
from clearml import Task
import numpy as np
import pandas as pd

def weasel_d(X_train, y_train, X_test, y_test):
    clf_WEASEL_V2 = WEASEL_V2(random_state=1379, n_jobs=4)
    clf_WEASEL_V2.fit(X_train, y_train)
    WEASEL_V2_pred = clf_WEASEL_V2.predict(X_test)

    return {
        'accuracy_score': accuracy_score(y_test, WEASEL_V2_pred), 
        'f1_score': f1_score(y_test, WEASEL_V2_pred, average='weighted'), 
        'precision_score': precision_score(y_test, WEASEL_V2_pred, average='weighted'), 
        'recall_score': recall_score(y_test, WEASEL_V2_pred, average='weighted'),
    }

def run_weasel_d(
    clearML=False,
    params={
        'k': 1,
        'K': 10,
        'country': 0,
        'city': 0,
        'category': None,
    },
    task=None,
    task_name="weasel_d",
):
    import time
    start_time = time.time()

    from classifiers.load_fold import load_fold

    if clearML:
        if task is None:
            task = Task.init(project_name='PopularTimesFold/Classifier', task_name="weasel_d")
        task.connect(params)

    # Carregar os dados
    df_timeseries = pd.read_csv('weekdays_datasets/df_timeseries.csv')
    df_signatures = pd.read_csv('weekdays_datasets/df_signatures.csv')

    # Realizar operações para combinar os dados
    # Exemplo: Calcular a média da assinatura para cada categoria
    signature_means = df_signatures.groupby('category').mean().reset_index()
    signature_means = signature_means.rename(columns={'signature_value': 'signature_mean'})  # Renomeie conforme necessário

    # Mesclar as assinaturas com o dataframe de séries temporais
    df_combined = df_timeseries.merge(signature_means, on='category', how='left')

    # Dividir os dados combinados em conjuntos de treino e teste
    name, X_train, y_train, X_test, y_test = load_fold(
        df_combined,
        params['k'],
        params['K'],
        params['country'],
        params['city'],
        params['category'],
    )
    
    print(f'Loaded: {name}')
    
    # Executa a função principal
    main_time = time.time()
    results = weasel_d(X_train, y_train, X_test, y_test)
    if clearML:
        task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - main_time)
        # Relatar resultados
        for key, value in results.items():
            task.get_logger().report_scalar('metrics', key, iteration=0, value=value)
        task.close()
    return results

if __name__ == '__main__':
    run_weasel_d()
