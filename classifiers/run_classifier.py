def run_classifier(fn, fn_name):
    import time
    start_time = time.time()

    import numpy as np

    from clearml import Task
    params = {
        'fold': 0,
        'data_task_name': 'example_data',
    }
    task = Task.init(project_name='PopularTimesFold/Classifier', task_name=fn_name)
    task.connect(params)

    preprocess_task = Task.get_task(project_name='PopularTimesFold/Data', task_name=params['data_task_name'])
    X_train = preprocess_task.artifacts['X_train'].get_local_copy()
    y_train = preprocess_task.artifacts['y_train'].get_local_copy()
    X_test = preprocess_task.artifacts['X_test'].get_local_copy()
    y_test = preprocess_task.artifacts['y_test'].get_local_copy()
    
    # Executes main function:
    main_time = time.time()
    results = fn(X_train, y_train, X_test, y_test)
    task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - main_time)

    # Reports results:
    for key, value in results.items():
        task.get_logger().report_scalar('metrics', key, iteration=0, value=value)
    task.get_logger().report_scalar('execution_time', 'total', iteration=0, value=time.time() - start_time)