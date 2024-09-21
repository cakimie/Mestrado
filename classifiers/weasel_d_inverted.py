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
    clearML=False,
    params={
        'k': 1,
        'K': 5,
        'country': 0, 
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

    df = pd.read_csv('weekdays_datasets/df_timeseries.csv')

    df_country_0 = df[df['country'] == 0] 
    df_country_1 = df[df['country'] == 1] 

    print(df_country_0)

    X_train = df_country_0.drop(columns=['category']) 
    y_train = df_country_0['category'] 

    X_test = df_country_1.drop(columns=['category']) 
    y_test = df_country_1['category'] 

    print(f'Treinando no país 0 e testando no país 1')
    results = weasel_d(X_train, y_train, X_test, y_test)

    print(f'Treinando no país 1 e testando no país 0')
    X_train = df_country_1.drop(columns=['category']) 
    y_train = df_country_1['category'] 

    X_test = df_country_0.drop(columns=['category']) 
    y_test = df_country_0['category'] 

    results_inverted = weasel_d(X_train, y_train, X_test, y_test)

    if clearML:
        task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - start_time)
        for key, value in results.items():
            task.get_logger().report_scalar('metrics', f'{key}_train_country_0_test_country_1', iteration=0, value=value)
        for key, value in results_inverted.items():
            task.get_logger().report_scalar('metrics', f'{key}_train_country_1_test_country_0', iteration=0, value=value)
        task.close()

    return results, results_inverted

if __name__ == '__main__':
    results, results_inverted = run_weasel_d()
    print(results)
    print(results_inverted)