from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier

def inception_time (X_train, y_train, X_test, y_test):  

    sns.set_style("whitegrid")
    np.unique(y_train)
    network = InceptionTimeClassifier(n_epochs=10000, verbose=False)
    network.fit(X_train, y_train)
    inception_pred = network.predict(X_test) 

    return {
        'accuracy_score': accuracy_score(y_test, inception_pred), 
        'f1_score': f1_score(y_test, inception_pred, average='weighted'), 
        'precision_score': precision_score(y_test, inception_pred, average='weighted'), 
        'recall_score': recall_score(y_test, inception_pred, average='weighted'),
    }