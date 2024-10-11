# RUN WITH python -m classifiers.hydra_ridge
# ATTENTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import torch
from hydra import Hydra, SparseScaler
from sklearn.linear_model import RidgeClassifierCV
from clearml import Task

def hydra_ridge (X_train, y_train, X_test, y_test):

    X_train_tensor = torch.tensor(np.expand_dims(X_train, axis=1), dtype=torch.float32)
    X_test_tensor = torch.tensor(np.expand_dims(X_test, axis=1), dtype=torch.float32)

    transform = Hydra(X_train.shape[-1])

    X_training_transform = transform(X_train_tensor)
    X_test_transform = transform(X_test_tensor)

    scaler = SparseScaler()

    X_training_transform = scaler.fit_transform(X_training_transform)
    X_test_transform = scaler.transform(X_test_transform)

    hydra_classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
    hydra_classifier.fit(X_training_transform, y_train)

    hydra_pred = hydra_classifier.predict(X_test_transform)

    return {
        'accuracy_score': accuracy_score(y_test, hydra_pred), 
        'f1_score': f1_score(y_test, hydra_pred, average='weighted'), 
        'precision_score': precision_score(y_test, hydra_pred, average='weighted'), 
        'recall_score': recall_score(y_test, hydra_pred, average='weighted'),
    }

def run_hydra_ridge(
    clearML = False,
    params = {
        'k': 1,
        'K': 10,
        'country': 0,
        'city': 0,
        'category': None,
    },
    task=None,
    task_name="hydra_ridge",
    dataset_filename=None,
):
    import time
    start_time = time.time()

    import numpy as np
    import pandas as pd
    from classifiers.load_fold import load_fold

    if clearML:
        if task==None:
            task = Task.init(project_name='PopularTimesFold/Classifier', task_name="hydra_ridge")
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
    main_time = time.time()
    results = hydra_ridge(X_train, y_train, X_test, y_test)
    if clearML:
        task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - main_time)
        # Reports results:
        for key, value in results.items():
            task.get_logger().report_scalar('metrics', key, iteration=0, value=value)
        task.close()
    return results

if __name__ == '__main__':
    run_hydra_ridge()