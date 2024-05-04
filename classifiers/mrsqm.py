from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from mrsqm import MrSQMClassifier, MrSQMTransformer

def MrSQM (X_train, y_train, X_test, y_test):

    clf_MrSQMC = MrSQMClassifier(nsax=0, nsfa=5)
    clf_MrSQMC.fit(X_train,y_train)

    MrSQM_pred = clf_MrSQMC.predict(X_test)

    return {
        'accuracy_score': accuracy_score(y_test, MrSQM_pred), 
        'f1_score': f1_score(y_test, MrSQM_pred, average='weighted'), 
        'precision_score': precision_score(y_test, MrSQM_pred, average='weighted'), 
        'recall_score': recall_score(y_test, MrSQM_pred, average='weighted'),
    }