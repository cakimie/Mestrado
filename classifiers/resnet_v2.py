from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sktime.classification.deep_learning.resnet import ResNetClassifier
from clearml import Task

def resnet(X_train, y_train, X_test, y_test):
    clf_resnet = ResNetClassifier(n_epochs=5000)
    clf_resnet.fit(X_train, y_train)
    resnet_pred = clf_resnet.predict(X_test) 

    return {
        'accuracy_score': accuracy_score(y_test, resnet_pred), 
        'f1_score': f1_score(y_test, resnet_pred, average='weighted'), 
        'precision_score': precision_score(y_test, resnet_pred, average='weighted'), 
        'recall_score': recall_score(y_test, resnet_pred, average='weighted'),
    }

def run_resnet(
    clearML=True,
    params={
        'k': 1,
        'K': 5,  # Number of folds for cross-validation
        'country': 0,
        'city': 0,
        'category': None,
    },
    task=None,
    task_name="resnet",
):
    import time
    import numpy as np
    import pandas as pd
    from classifiers.load_fold import load_fold

    if clearML:
        if task is None:
            task = Task.init(project_name='PopularTimesFold/Classifier', task_name=task_name)
        task.connect(params)

    df = pd.read_csv('weekdays_datasets/df_timeseries.csv')

    # Metrics dictionary to accumulate results across folds
    metrics = {
        'accuracy_score': [],
        'f1_score': [],
        'precision_score': [],
        'recall_score': [],
    }

    # Loop over each fold (1 to K)
    for fold in range(1, params['K'] + 1):
        print(f'Running fold {fold}')
        name, X_train, y_train, X_test, y_test = load_fold(
            df,
            fold,
            params['K'],
            params['country'],
            params['city'],
            params['category'],
        )
        print(f'Loaded: {name}')
        
        # Execute the ResNet model
        main_time = time.time()
        results = resnet(X_train, y_train, X_test, y_test)

        # Accumulate the results for each metric
        for key in metrics.keys():
            metrics[key].append(results[key])
        
        if clearML:
            task.get_logger().report_scalar('execution_time', 'fold', iteration=fold, value=time.time() - main_time)
            for key, value in results.items():
                task.get_logger().report_scalar('metrics', key, iteration=fold, value=value)

    # Calculate the average results across all folds
    avg_metrics = {key: np.mean(value) for key, value in metrics.items()}

    if clearML:
        for key, value in avg_metrics.items():
            task.get_logger().report_scalar('metrics', key, iteration=0, value=value)
        task.close()

    return avg_metrics

if __name__ == '__main__':
    results = run_resnet()
    print('5-Fold Cross-Validation Results:', results)