from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from aeon.classification.shapelet_based import RDSTClassifier
from clearml import Task

def rdst (X_train, y_train, X_test, y_test):

    clf_rdst = RDSTClassifier()
    clf_rdst.fit(X_train, y_train)
    rdst_pred = clf_rdst.predict(X_test)
    
    return {
        'accuracy_score': accuracy_score(y_test, rdst_pred), 
        'f1_score': f1_score(y_test, rdst_pred, average='weighted'), 
        'precision_score': precision_score(y_test, rdst_pred, average='weighted'), 
        'recall_score': recall_score(y_test, rdst_pred, average='weighted'),
    }

def run_rdst(
    clearML = False,
    params = {
        'k': 1,
        'K': 10,
        'country': 0,
        'city': 0,
        'category': None,
    },
    task=None,
    task_name="rdst",
    dataset_filename=None,
):
    import time
    start_time = time.time()

    import numpy as np
    import pandas as pd
    from classifiers.load_fold import load_fold

    if clearML:
        if task==None:
            task = Task.init(project_name='PopularTimesFold/Classifier', task_name="rdst")
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
    print(df.columns)
    
    # Executes main function:
    main_time = time.time()
    results = rdst(X_train, y_train, X_test, y_test)
    if clearML:
        task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - main_time)
        # Reports results:
        for key, value in results.items():
            task.get_logger().report_scalar('metrics', key, iteration=0, value=value)
        task.close()
    return results

if __name__ == '__main__':
    run_rdst()