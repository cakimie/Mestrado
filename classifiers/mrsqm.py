# RUN WITH python -m classifiers.mrsqm 
# ATTENTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from mrsqm import MrSQMClassifier, MrSQMTransformer

def MrSQM (X_train, y_train, X_test, y_test):

    clf_MrSQMC = MrSQMClassifier(nsax=0, nsfa=5)
    clf_MrSQMC.fit(X_train,y_train)

    MrSQM_pred = clf_MrSQMC.predict(X_test)

    return {
        'accuracy_score': accuracy_score(y_test, MrSQM_pred), 
        'f1_score': f1_score(y_test, MrSQM_pred, average='weighted'), 
        'precision_score': precision_score(y_test, MrSQM_pred, average='weighted'), 
        'recall_score': recall_score(y_test, MrSQM_pred, average='weighted'),
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
    task = Task.init(project_name='PopularTimesFold/Classifier', task_name="MrSQM")
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
    results = MrSQM(X_train, y_train, X_test, y_test)
    task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - main_time)

    # Reports results:
    for key, value in results.items():
        task.get_logger().report_scalar('metrics', key, iteration=0, value=value)
    task.get_logger().report_scalar('execution_time', 'total', iteration=0, value=time.time() - start_time)