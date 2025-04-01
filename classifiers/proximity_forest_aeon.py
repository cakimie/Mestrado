from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from aeon.classification.distance_based import ProximityForest
from clearml import Task

def proximity_forest(X_train, y_train, X_test, y_test):
    clf_pf = ProximityForest(
        n_trees=20, n_splitters=5, max_depth=10
    ) 
    clf_pf.fit(X_train, y_train) 
    pf_pred = clf_pf.predict(X_test) 

    return {
        'accuracy_score': accuracy_score(y_test, pf_pred), 
        'f1_score': f1_score(y_test, pf_pred, average='weighted'), 
        'precision_score': precision_score(y_test, pf_pred, average='weighted'), 
        'recall_score': recall_score(y_test, pf_pred, average='weighted'),
    }

def run_proximity_forest(
    clearML=False,
    params={
        'k': 1,
        'K': 5,  # Number of folds
        'country': 0,
        'city': 0,
        'category': None,
    },
    task=None,
    task_name="proximity_forest",
):
    import time
    start_time = time.time()

    import numpy as np
    import pandas as pd
    from classifiers.load_fold import load_fold
    import sys

    sys.setrecursionlimit(30000)

    if clearML:
        if task is None:
            task = Task.init(project_name='PopularTimesFold/Classifier', task_name=task_name)
        task.connect(params)

    df = pd.read_csv('weekdays_datasets/df_timeseries.csv')

    # Metrics dictionary to accumulate results
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
        
        # Execute main function
        main_time = time.time()
        results = proximity_forest(X_train, y_train, X_test, y_test)

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
    results = run_proximity_forest()
    print('5-Fold Cross-Validation Results:', results)
