import os
import numpy as np
import pandas as pd
import time
from sktime.classification.distance_based import ProximityForest
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy import stats

# Function to apply filtering, k-fold validation, and collect metrics
def run_filtered_kfold(df, filter_column='city', n_splits=5):
    unique_values = df[filter_column].unique()
    all_metrics = []

    # Loop over each unique value in the filter column (e.g., each city)
    for value in unique_values:
        print(f"Processing {filter_column} = {value}")

        # Filter dataset for the current value
        subset_df = df[df[filter_column] == value]
        X = subset_df.drop(columns=['category', 'country', filter_column]).values  # Ensure category and filter_column are removed
        y = subset_df['category'].values

        # Initialize metrics collection for this subset
        accuracy_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        start_time = time.time()  # Track execution time

        # Perform 5-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        clf_pf = ProximityForest(n_estimators=2, max_depth=2, n_stump_evaluations=1)
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Fit and predict
            clf_pf.fit(X_train, y_train)
            y_pred_pf = clf_pf.predict(X_test)
            
            # Append fold metrics
            accuracy_scores.append(accuracy_score(y_test, y_pred_pf))
            f1_scores.append(f1_score(y_test, y_pred_pf, average='weighted'))
            precision_scores.append(precision_score(y_test, y_pred_pf, average='weighted'))
            recall_scores.append(recall_score(y_test, y_pred_pf, average='weighted'))

        end_time = time.time()
        execution_time = end_time - start_time

        # Calculate mean and std for each metric within this subset
        metrics = {
            "filter_value": value,
            "execution_time": execution_time,
            "accuracy_mean": np.mean(accuracy_scores),
            "accuracy_std": np.std(accuracy_scores),
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
            "precision_mean": np.mean(precision_scores),
            "precision_std": np.std(precision_scores),
            "recall_mean": np.mean(recall_scores),
            "recall_std": np.std(recall_scores)
        }
        
        # Calculate confidence intervals
        confidence_level = 0.95
        t_score = stats.t.ppf((1 + confidence_level) / 2, n_splits - 1)
        
        metrics["accuracy_ci"] = (metrics["accuracy_mean"] - t_score * metrics["accuracy_std"] / np.sqrt(n_splits),
                                  metrics["accuracy_mean"] + t_score * metrics["accuracy_std"] / np.sqrt(n_splits))
        metrics["f1_ci"] = (metrics["f1_mean"] - t_score * metrics["f1_std"] / np.sqrt(n_splits),
                            metrics["f1_mean"] + t_score * metrics["f1_std"] / np.sqrt(n_splits))
        metrics["precision_ci"] = (metrics["precision_mean"] - t_score * metrics["precision_std"] / np.sqrt(n_splits),
                                   metrics["precision_mean"] + t_score * metrics["precision_std"] / np.sqrt(n_splits))
        metrics["recall_ci"] = (metrics["recall_mean"] - t_score * metrics["recall_std"] / np.sqrt(n_splits),
                                metrics["recall_mean"] + t_score * metrics["recall_std"] / np.sqrt(n_splits))
        
        # Append metrics for this subset to all metrics
        all_metrics.append(metrics)

    # Calculate overall mean metrics across all filter values
    overall_metrics = pd.DataFrame(all_metrics).mean(numeric_only=True).to_dict()
    overall_metrics["filter_value"] = "Overall"

    # Append overall metrics to the list
    all_metrics.append(overall_metrics)

    # Create a DataFrame from all metrics and save to CSV
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv("results_proximity_forest_5fold_city.csv", index=False)
    print("Results saved to results_proximity_forest_5fold_city.csv")

# Load the dataset
file_path = 'weekdays_datasets/df_timeseries.csv'
df = pd.read_csv(file_path)

# Run the filtered k-fold validation
run_filtered_kfold(df, filter_column='city')