from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble

from clearml import Task

def tde (X_train, y_train, X_test, y_test):

    clf_TDE = TemporalDictionaryEnsemble(
        n_parameter_samples=250,
        max_ensemble_size=50,
        randomly_selected_params=50,
    ) 
    clf_TDE.fit(X_train, y_train) 
    tde_pred = clf_TDE.predict(X_test)

    return {
        'accuracy_score': accuracy_score(y_test, tde_pred), 
        'f1_score': f1_score(y_test, tde_pred, average='weighted'), 
        'precision_score': precision_score(y_test, tde_pred, average='weighted'), 
        'recall_score': recall_score(y_test, tde_pred, average='weighted'),
    }

def run_tde(
    clearML = False,
    params = {
        'k': 1,
        'K': 5,
        'country': 0,
        'city': 0,
        'category': None,
    },
    task=None,
    task_name="tde",
    dataset_filename=None,
):
    import time
    start_time = time.time()

    import numpy as np
    import pandas as pd
    from classifiers.load_fold import load_fold

    if clearML:
        if task==None:
            task = Task.init(project_name='PopularTimesFold/Classifier', task_name="tde")
        task.connect(params)

    if dataset_filename:
        df = pd.read_csv(dataset_filename)
    else:
        df = pd.read_csv('weekdays_datasets/df_timeseries.csv')

    name, X_train, y_train, X_test, y_test = load_fold(
        df,
        params['k'],
        params['K'],
        params['country'],
        params['city'],
        params['category'],
    )
    print(f'Loaded: {name}')
    
    # Executes main function:
    df_country_0 = df[df['country'] == 0] 
    df_country_1 = df[df['country'] == 1] 

    print(df_country_0)

    X_train = np.array(df_country_0.drop(columns=['category', 'id']))
    y_train = np.array(df_country_0['category'])

    X_test = np.array(df_country_1.drop(columns=['category', 'id']))
    y_test = np.array(df_country_1['category'])

    print(f'Treinando no país 0 e testando no país 1')
    results = tde(X_train, y_train, X_test, y_test)

    print(f'Treinando no país 1 e testando no país 0')
    X_train = np.array(df_country_1.drop(columns=['category', 'id']))
    y_train = np.array(df_country_1['category'])

    X_test = np.array(df_country_0.drop(columns=['category', 'id']))
    y_test = np.array(df_country_0['category'])

    results_inverted = tde(X_train, y_train, X_test, y_test)

    if clearML:
        task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - main_time)
        # Reports results:
        for key, value in results.items():
            task.get_logger().report_scalar('metrics', key, iteration=0, value=value)
        task.close()
    return results, results_inverted

if __name__ == '__main__':
    results, results_inverted = run_tde()
    print(results)
    print(results_inverted)