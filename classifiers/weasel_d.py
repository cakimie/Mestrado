from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from weasel.classification.dictionary_based import WEASEL_V2

def weasel_d (X_train, y_train, X_test, y_test):

    clf_WEASEL_V2 = WEASEL_V2(random_state=1379, n_jobs=4)
    clf_WEASEL_V2.fit(X_train,y_train)
    WEASEL_V2_pred = clf_WEASEL_V2.predict(X_test)

    return {
        'accuracy_score': accuracy_score(y_test, WEASEL_V2_pred), 
        'f1_score': f1_score(y_test, WEASEL_V2_pred, average='weighted'), 
        'precision_score': precision_score(y_test, WEASEL_V2_pred, average='weighted'), 
        'recall_score': recall_score(y_test, WEASEL_V2_pred, average='weighted'),
    }

if __name__ == '__main__':
    from classifiers.run_classifier import run_classifier
    run_classifier(weasel_d, 'weasel_d')