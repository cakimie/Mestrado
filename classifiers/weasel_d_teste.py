from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from weasel.classification.dictionary_based import WEASEL_V2
from clearml import Task
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
    clearML=True,
    params={
        'k': 1,
        'K': 10,
        'country': 0,  # Não será mais necessário este parâmetro
        'city': 0,
        'category': None,
    },
    task=None,
    task_name="weasel_d",
):
    import time
    start_time = time.time()

    if clearML:
        if task is None:
            task = Task.init(project_name='PopularTimesFold/Classifier', task_name=task_name)
        task.connect(params)

    # Carregar o dataset
    df = pd.read_csv('weekdays_datasets/df_timeseries.csv')

    # Separar os dados por país
    df_country_0 = df[df['country'] == 0]  # Dados do país 0
    df_country_1 = df[df['country'] == 1]  # Dados do país 1

    # Assumimos que as colunas X (features) estão no DataFrame e y é a coluna de rótulo (supondo que se chama 'label')
    X_train = df_country_0.drop(columns=['category'])  # Features do país 0
    y_train = df_country_0['category']  # Rótulos do país 0

    X_test = df_country_1.drop(columns=['category'])  # Features do país 1
    y_test = df_country_1['category']  # Rótulos do país 1

    # Treinar no país 0 e testar no país 1
    print(f'Treinando no país 0 e testando no país 1')
    results = weasel_d(X_train, y_train, X_test, y_test)

    # Agora inverte, treinando no país 1 e testando no país 0
    print(f'Treinando no país 1 e testando no país 0')
    X_train = df_country_1.drop(columns=['category'])  # Features do país 1
    y_train = df_country_1['category']  # Rótulos do país 1

    X_test = df_country_0.drop(columns=['category'])  # Features do país 0
    y_test = df_country_0['category']  # Rótulos do país 0

    results_inverted = weasel_d(X_train, y_train, X_test, y_test)

    # Relatar resultados no ClearML
    if clearML:
        task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - start_time)
        for key, value in results.items():
            task.get_logger().report_scalar('metrics', f'{key}_train_country_0_test_country_1', iteration=0, value=value)
        for key, value in results_inverted.items():
            task.get_logger().report_scalar('metrics', f'{key}_train_country_1_test_country_0', iteration=0, value=value)
        task.close()

    return results, results_inverted

if __name__ == '__main__':
    run_weasel_d()
