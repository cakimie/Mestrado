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

if __name__ == '__main__':
    from run_classifier import run_classifier
    run_classifier(ridge_cv, 'ridge_cv')