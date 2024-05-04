import numpy as np
from clearml import Task

task = Task.init(project_name='PopularTimesFold/Data', task_name='example_data')

X_train = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])
print(f'X_train.shape={X_train.shape}')
task.upload_artifact(name='X_train', artifact_object=X_train)

y_train = np.array([0, 0, 1, 1],)
print(f'y_train.shape={y_train.shape}')
task.upload_artifact(name='y_train', artifact_object=y_train)

task.upload_artifact(name='X_test', artifact_object=X_train)
task.upload_artifact(name='y_test', artifact_object=y_train)