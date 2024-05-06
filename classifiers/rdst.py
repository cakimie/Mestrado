from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from aeon.classification.shapelet_based import RDSTClassifier

def rdst (X_train, y_train, X_test, y_test):

    clf_rdst = RDSTClassifier()
    clf_rdst.fit(X_train, y_train)
    rdst_pred = clf_rdst.predict(X_test)
    
    return {
        'accuracy_score': accuracy_score(y_test, rdst_pred), 
        'f1_score': f1_score(y_test, rdst_pred, average='weighted'), 
        'precision_score': precision_score(y_test, rdst_pred, average='weighted'), 
        'recall_score': recall_score(y_test, rdst_pred, average='weighted'),
    }

if __name__ == '__main__':
    from classifiers.run_classifier import run_classifier
    run_classifier(rdst, 'rdst')