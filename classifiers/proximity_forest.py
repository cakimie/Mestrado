from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sktime.classification.distance_based import ProximityForest

def proximity_forest (X_train, y_train, X_test, y_test):

    clf_pf = ProximityForest(
        n_estimators=2, max_depth=2, n_stump_evaluations=1
    ) 
    clf_pf.fit(X_train, y_train) 
    pf_pred = clf_pf.predict(X_test) 

    return {
        'accuracy_score': accuracy_score(y_test, pf_pred), 
        'f1_score': f1_score(y_test, pf_pred, average='weighted'), 
        'precision_score': precision_score(y_test, pf_pred, average='weighted'), 
        'recall_score': recall_score(y_test, pf_pred, average='weighted'),
    }

if __name__ == '__main__':
    from run_classifier import run_classifier
    run_classifier(proximity_forest, 'proximity_forest')