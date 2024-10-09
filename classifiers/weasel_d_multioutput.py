from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from weasel.classification.dictionary_based import WEASEL_V2
from sklearn.multioutput import MultiOutputClassifier
from clearml import Task

# Função que treina e avalia o classificador WEASEL_V2 com MultiOutputClassifier
def weasel_d(X_train, y_train, X_test, y_test):
    # Envolvendo o WEASEL_V2 com MultiOutputClassifier para múltiplas saídas
    clf_WEASEL_V2 = MultiOutputClassifier(WEASEL_V2(random_state=1379, n_jobs=4))
    clf_WEASEL_V2.fit(X_train, y_train)
    WEASEL_V2_pred = clf_WEASEL_V2.predict(X_test)

    # Calculando métricas para cada saída (multi-output)
    results = {
        'accuracy_score': accuracy_score(y_test, WEASEL_V2_pred), 
        'f1_score': f1_score(y_test, WEASEL_V2_pred, average='weighted'), 
        'precision_score': precision_score(y_test, WEASEL_V2_pred, average='weighted'), 
        'recall_score': recall_score(y_test, WEASEL_V2_pred, average='weighted'),
    }

    return results

# Função principal
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

    import numpy as np
    import pandas as pd
    from classifiers.load_fold import load_fold

    if clearML:
        if task is None:
            task = Task.init(project_name='PopularTimesFold/Classifier', task_name="weasel_d")
        task.connect(params)

    df = pd.read_csv('weekdays_datasets/df_timeseries.csv')
    
    # Carregar o dataset para múltiplas colunas-alvo
    name, X_train, y_train, X_test, y_test = load_fold(
        df,
        params['k'],
        params['K'],
        params['country'],
        params['city'],
        params['category'],
    )
    
    # Certifique-se de que y_train e y_test sejam arrays com múltiplas colunas
    # para country, city, category, se necessário.
    print(f'Loaded: {name}')
    
    # Executa a função principal
    main_time = time.time()
    results = weasel_d(X_train, y_train, X_test, y_test)
    
    # Relatar resultados se o ClearML estiver ativo
    if clearML:
        task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - main_time)
        for key, value in results.items():
            task.get_logger().report_scalar('metrics', key, iteration=0, value=value)
        task.close()
    
    return results

if __name__ == '__main__':
    run_weasel_d()