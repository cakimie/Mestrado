from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sktime.classification.feature_based import FreshPRINCE

def fresh_prince (X_train, y_train, X_test, y_test):

    clf_freshPRINCE = FreshPRINCE(
    default_fc_parameters="minimal",
    n_estimators=5,
    )
    clf_freshPRINCE.fit(X_train, y_train)
    fp_pred = clf_freshPRINCE.predict(X_test)

    return {
        'accuracy_score': accuracy_score(y_test, fp_pred), 
        'f1_score': f1_score(y_test, fp_pred, average='weighted'), 
        'precision_score': precision_score(y_test, fp_pred, average='weighted'), 
        'recall_score': recall_score(y_test, fp_pred, average='weighted'),
    }

if __name__ == '__main__':
    from run_classifier import run_classifier
    run_classifier(fresh_prince, 'fresh_prince')