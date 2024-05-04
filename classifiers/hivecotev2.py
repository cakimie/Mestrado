from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.sklearn import RotationForest

def hivecotev2 (X_train, y_train, X_test, y_test):
    hc2_classifier = HIVECOTEV2(
        stc_params={
            "estimator": RotationForest(n_estimators=3),
            "n_shapelet_samples": 100,
            "max_shapelets": 10,
            "batch_size": 20,
        },
        drcif_params={"n_estimators": 2, "n_intervals": 2, "att_subsample_size": 2},
        arsenal_params={"num_kernels": 50, "n_estimators": 3},
        tde_params={
            "n_parameter_samples": 10,
            "max_ensemble_size": 3,
            "randomly_selected_params": 5,
        },
    )
    hc2_classifier.fit(X_train, y_train)
    hc2_pred = hc2_classifier.predict(X_test)

    return {
        'accuracy_score': accuracy_score(y_test, hc2_pred), 
        'f1_score': f1_score(y_test, hc2_pred, average='weighted'), 
        'precision_score': precision_score(y_test, hc2_pred, average='weighted'), 
        'recall_score': recall_score(y_test, hc2_pred, average='weighted'),
    }

if __name__ == '__main__':
    from run_classifier import run_classifier
    run_classifier(hivecotev2, 'hivecotev2')