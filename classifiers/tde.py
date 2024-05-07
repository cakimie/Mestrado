from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble

def tde (X_train, y_train, X_test, y_test):

    clf_TDE = TemporalDictionaryEnsemble(
        n_parameter_samples=10,
        max_ensemble_size=3,
        randomly_selected_params=5,
    ) 
    clf_TDE.fit(X_train, y_train) 
    tde_pred = clf_TDE.predict(X_test)

    return {
        'accuracy_score': accuracy_score(y_test, tde_pred), 
        'f1_score': f1_score(y_test, tde_pred, average='weighted'), 
        'precision_score': precision_score(y_test, tde_pred, average='weighted'), 
        'recall_score': recall_score(y_test, tde_pred, average='weighted'),
    }

if __name__ == '__main__':
    import time
    start_time = time.time()

    import numpy as np
    import pandas as pd
    from classifiers.load_fold import load_fold

    from clearml import Task
    params = {
        'k': 1,
        'K': 10,
        'country': 0,
        'city': 0,
        'category': None,
    }
    task = Task.init(project_name='PopularTimesFold/Classifier', task_name="tde")
    task.connect(params)

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
    main_time = time.time()
    results = tde(X_train, y_train, X_test, y_test)
    task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - main_time)

    # Reports results:
    for key, value in results.items():
        task.get_logger().report_scalar('metrics', key, iteration=0, value=value)
    task.get_logger().report_scalar('execution_time', 'total', iteration=0, value=time.time() - start_time)