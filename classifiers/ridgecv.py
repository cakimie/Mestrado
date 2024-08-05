# RUN WITH python -m classifiers.ridgecv
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from mrsqm import MrSQMTransformer

def ridge_cv (X_train, y_train, X_test, y_test):

    tfm = MrSQMTransformer()
    X_train_transform = tfm.fit_transform(X_train,y_train)
    X_test_transform = tfm.transform(X_test)

    # use ridgecv classifier
    ridge = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10)).fit(X_train_transform,y_train)
    ridgecvt_pred = ridge.predict(X_test_transform)

    return {
        'accuracy_score': accuracy_score(y_test, ridgecvt_pred), 
        'f1_score': f1_score(y_test, ridgecvt_pred, average='weighted'), 
        'precision_score': precision_score(y_test, ridgecvt_pred, average='weighted'), 
        'recall_score': recall_score(y_test, ridgecvt_pred, average='weighted'),
    }

def run_ridge_cv(
    clearML = True,
    params = {
        'k': 1,
        'K': 10,
        'country': 0,
        'city': 0,
        'category': None,
    },
    task=None,
    task_name="ridge_cv",
):
    import time
    start_time = time.time()

    import numpy as np
    import pandas as pd
    from classifiers.load_fold import load_fold

    if clearML:
        if task==None:
            task = Task.init(project_name='PopularTimesFold/Classifier', task_name="ridge_cv")
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
    results = ridge_cv(X_train, y_train, X_test, y_test)
    if clearML:
        task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - main_time)
        # Reports results:
        for key, value in results.items():
            task.get_logger().report_scalar('metrics', key, iteration=0, value=value)
        task.close()
    return results

if __name__ == '__main__':
    run_ridge_cv()