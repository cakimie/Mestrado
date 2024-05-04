from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sktime.classification.deep_learning.resnet import ResNetClassifier

def resnet (X_train, y_train, X_test, y_test):

    clf_resnet = ResNetClassifier(n_epochs=10000) 
    clf_resnet.fit(X_train, y_train)
    resnet_pred = clf_resnet.predict(X_test) 

    return {
        'accuracy_score': accuracy_score(y_test, resnet_pred), 
        'f1_score': f1_score(y_test, resnet_pred, average='weighted'), 
        'precision_score': precision_score(y_test, resnet_pred, average='weighted'), 
        'recall_score': recall_score(y_test, resnet_pred, average='weighted'),
    }