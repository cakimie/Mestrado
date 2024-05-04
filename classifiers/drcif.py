from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from aeon.classification.compose import ChannelEnsembleClassifier
from aeon.classification.interval_based import DrCIFClassifier


def DrCIF (X_train, y_train, X_test, y_test):

    cls_DrCIF = ChannelEnsembleClassifier(
        estimators=[
            ("DrCIF0", DrCIFClassifier(n_estimators=5, n_intervals=2), [0]),
            #("ROCKET3", RocketClassifier(num_kernels=1000), [3, 4]),
        ]
    )

    cls_DrCIF.fit(X_train, y_train)
    DrCIF_pred = cls_DrCIF.predict(X_test)

    return {
        'accuracy_score': accuracy_score(y_test, DrCIF_pred), 
        'f1_score': f1_score(y_test, DrCIF_pred, average='weighted'), 
        'precision_score': precision_score(y_test, DrCIF_pred, average='weighted'), 
        'recall_score': recall_score(y_test, DrCIF_pred, average='weighted'),
    }

if __name__ == '__main__':
    from run_classifier import run_classifier
    run_classifier(DrCIF, 'DrCIF')