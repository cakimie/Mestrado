from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from aeon.classification.distance_based import ElasticEnsemble

def elastic_ensemble (X_train, y_train, X_test, y_test):

    ee_clf = ElasticEnsemble(
    proportion_of_param_options=0.1,
    proportion_train_for_test=0.1,
    distance_measures = ["dtw","ddtw"],
    majority_vote=True,
    )
    ee_clf.fit(X_train, y_train)

    ee_pred = ee_clf.predict(X_test)

    return {
        'accuracy_score': accuracy_score(y_test, ee_pred), 
        'f1_score': f1_score(y_test, ee_pred, average='weighted'), 
        'precision_score': precision_score(y_test, ee_pred, average='weighted'), 
        'recall_score': recall_score(y_test, ee_pred, average='weighted'),
    }