from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sktime.classification.feature_based import TSFreshClassifier
from sklearn.ensemble import RandomForestClassifier

def ts_fresh (X_train, y_train, X_test, y_test):

    clf_TSFresh = TSFreshClassifier(
    default_fc_parameters="minimal",
    estimator=RandomForestClassifier(n_estimators=5),
    )
    clf_TSFresh.fit(X_train, y_train)
    TSFreshClassifier(...)
    tsfresh_pred = clf_TSFresh.predict(X_test)

    return {
        'accuracy_score': accuracy_score(y_test, tsfresh_pred), 
        'f1_score': f1_score(y_test, tsfresh_pred, average='weighted'), 
        'precision_score': precision_score(y_test, tsfresh_pred, average='weighted'), 
        'recall_score': recall_score(y_test, tsfresh_pred, average='weighted'),
    }

if __name__ == '__main__':
    from run_classifier import run_classifier
    run_classifier(ts_fresh, 'ts_fresh')