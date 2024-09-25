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
        'K': 5,
        'country': 0,
        'city': 0,
        'category': None,
    },
    task=None,
    task_name="hydra_ridge",
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

    df = pd.read_csv('weekdays_datasets/df_timeseries.csv')
    
    df_country_0 = df[df['country'] == 0] 
    df_country_1 = df[df['country'] == 1] 

    print(df_country_0)

    X_train = df_country_0.drop(columns=['category']) 
    y_train = df_country_0['category'] 

    X_test = df_country_1.drop(columns=['category']) 
    y_test = df_country_1['category'] 

    print(f'Treinando no país 0 e testando no país 1')
    results = hydra_ridge(X_train, y_train, X_test, y_test)

    print(f'Treinando no país 1 e testando no país 0')
    X_train = df_country_1.drop(columns=['category']) 
    y_train = df_country_1['category'] 

    X_test = df_country_0.drop(columns=['category']) 
    y_test = df_country_0['category'] 

    results_inverted = hydra_ridge(X_train, y_train, X_test, y_test)

    if clearML:
        task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - start_time)
        for key, value in results.items():
            task.get_logger().report_scalar('metrics', f'{key}_train_country_0_test_country_1', iteration=0, value=value)
        for key, value in results_inverted.items():
            task.get_logger().report_scalar('metrics', f'{key}_train_country_1_test_country_0', iteration=0, value=value)
        task.close()

    return results, results_inverted

if __name__ == '__main__':
    results, results_inverted = run_hydra_ridge()
    print(results)
    print(results_inverted)