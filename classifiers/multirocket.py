from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MultiRocket')))
print(sys.path)
from multirocket.multirocket import MultiRocket
from sklearn.preprocessing import LabelEncoder

def multirocket (X_train, y_train, X_test, y_test):

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    # returns ntc format, remove the last dimension
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    mrocket_classifier = MultiRocket(
        num_features=50000,
        classifier="logistic",
        verbose=2,
    )
    yhat_train = mrocket_classifier.fit(
        X_train, y_train,
        predict_on_train=False
    )
    mrocket_pred = mrocket_classifier.predict(X_test)

    return {
        'accuracy_score': accuracy_score(y_test, mrocket_pred), 
        'f1_score': f1_score(y_test, mrocket_pred, average='weighted'), 
        'precision_score': precision_score(y_test, mrocket_pred, average='weighted'), 
        'recall_score': recall_score(y_test, mrocket_pred, average='weighted'),
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
    task = Task.init(project_name='PopularTimesFold/Classifier', task_name="multirocket")
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
    results = multirocket(X_train, y_train, X_test, y_test)
    task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - main_time)

    # Reports results:
    for key, value in results.items():
        task.get_logger().report_scalar('metrics', key, iteration=0, value=value)
    task.get_logger().report_scalar('execution_time', 'total', iteration=0, value=time.time() - start_time)