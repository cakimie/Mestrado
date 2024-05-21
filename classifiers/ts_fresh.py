from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sktime.classification.feature_based import TSFreshClassifier
from sklearn.ensemble import RandomForestClassifier

from clearml import Task

def ts_fresh (X_train, y_train, X_test, y_test):

    clf_TSFresh = TSFreshClassifier(
    default_fc_parameters="comprehensive",
    estimator=RandomForestClassifier(n_estimators=10),
    )
    clf_TSFresh.fit(X_train, y_train)
    tsfresh_pred = clf_TSFresh.predict(X_test)

    return {
        'accuracy_score': accuracy_score(y_test, tsfresh_pred), 
        'f1_score': f1_score(y_test, tsfresh_pred, average='weighted'), 
        'precision_score': precision_score(y_test, tsfresh_pred, average='weighted'), 
        'recall_score': recall_score(y_test, tsfresh_pred, average='weighted'),
    }

def run_ts_fresh(
    clearML = True,
    params = {
        'k': 1,
        'K': 10,
        'country': 0,
        'city': 0,
        'category': None,
    },
    task=None,
    task_name="ts_fresh",
):
    import time
    start_time = time.time()

    import numpy as np
    import pandas as pd
    from classifiers.load_fold import load_fold

    if clearML:
        if task==None:
            task=Task.init(project_name='PopularTimesFold/Classifier', task_name=task_name, reuse_last_task_id=False)
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
    results = ts_fresh(X_train, y_train, X_test, y_test)
    if clearML:
        task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - main_time)
        task.close()
    return results

if __name__ == '__main__':
    run_ts_fresh()