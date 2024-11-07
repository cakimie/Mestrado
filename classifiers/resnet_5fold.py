import numpy as np
import pandas as pd
import time
from sktime.classification.deep_learning.resnet import ResNetClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy import stats

file_path = 'weekdays_datasets/df_timeseries.csv'
df = pd.read_csv(file_path)

X = df.drop(columns=['category','country','city']).values
y = df['category'].values

clf_resnet = ResNetClassifier(n_epochs=10000)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

accuracy_scores = []
f1_scores = []
precision_scores = []
recall_scores = []

start_time = time.time()

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    

    clf_resnet.fit(X_train, y_train)
    

    y_pred_resnet = clf_resnet.predict(X_test)
    

    accuracy_scores.append(accuracy_score(y_test, y_pred_resnet))
    f1_scores.append(f1_score(y_test, y_pred_resnet, average='weighted'))
    precision_scores.append(precision_score(y_test, y_pred_resnet, average='weighted'))
    recall_scores.append(recall_score(y_test, y_pred_resnet, average='weighted'))

end_time = time.time()
execution_time = end_time - start_time

accuracy_mean, accuracy_std = np.mean(accuracy_scores), np.std(accuracy_scores)
f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
precision_mean, precision_std = np.mean(precision_scores), np.std(precision_scores)
recall_mean, recall_std = np.mean(recall_scores), np.std(recall_scores)

confidence_level = 0.95
t_score = stats.t.ppf((1 + confidence_level) / 2, n_splits - 1)

accuracy_ci = (accuracy_mean - t_score * accuracy_std / np.sqrt(n_splits),
               accuracy_mean + t_score * accuracy_std / np.sqrt(n_splits))
f1_ci = (f1_mean - t_score * f1_std / np.sqrt(n_splits),
         f1_mean + t_score * f1_std / np.sqrt(n_splits))
precision_ci = (precision_mean - t_score * precision_std / np.sqrt(n_splits),
                precision_mean + t_score * precision_std / np.sqrt(n_splits))
recall_ci = (recall_mean - t_score * recall_std / np.sqrt(n_splits),
             recall_mean + t_score * recall_std / np.sqrt(n_splits))

print("Execution Time (s): {:.2f}".format(execution_time))
print("Accuracy: {:.4f} ± {:.4f}, CI: {}".format(accuracy_mean, accuracy_std, accuracy_ci))
print("F1-Score: {:.4f} ± {:.4f}, CI: {}".format(f1_mean, f1_std, f1_ci))
print("Precision: {:.4f} ± {:.4f}, CI: {}".format(precision_mean, precision_std, precision_ci))
print("Recall: {:.4f} ± {:.4f}, CI: {}".format(recall_mean, recall_std, recall_ci))

results = {
    "execution_time": execution_time,
    "accuracy_mean": accuracy_mean,
    "accuracy_std": accuracy_std,
    "accuracy_ci": accuracy_ci,
    "f1_mean": f1_mean,
    "f1_std": f1_std,
    "f1_ci": f1_ci,
    "precision_mean": precision_mean,
    "precision_std": precision_std,
    "precision_ci": precision_ci,
    "recall_mean": recall_mean,
    "recall_std": recall_std,
    "recall_ci": recall_ci,
}

results_df = pd.DataFrame([results])
results_df.to_csv("results_resnet5fold_summary.csv", index=False)
print("Results saved to results_resnet5fold_summary.csv")
