from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import torch
from hydra import Hydra, SparseScaler
from sklearn.linear_model import RidgeClassifierCV

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

if __name__ == '__main__':
    from run_classifier import run_classifier
    run_classifier(hydra_ridge, 'hydra_ridge')