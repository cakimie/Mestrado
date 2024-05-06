from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble

def tde (X_train, y_train, X_test, y_test):

    clf_TDE = TemporalDictionaryEnsemble(
        n_parameter_samples=10,
        max_ensemble_size=3,
        randomly_selected_params=5,
    ) 
    clf_TDE.fit(X_train, y_train) 
    tde_pred = clf_TDE.predict(X_test)

    return {
        'accuracy_score': accuracy_score(y_test, tde_pred), 
        'f1_score': f1_score(y_test, tde_pred, average='weighted'), 
        'precision_score': precision_score(y_test, tde_pred, average='weighted'), 
        'recall_score': recall_score(y_test, tde_pred, average='weighted'),
    }

if __name__ == '__main__':
    from classifiers.run_classifier import run_classifier
    run_classifier(tde, 'tde')