import argparse
import joblib
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, auc, f1_score, roc_auc_score, precision_recall_curve, classification_report

def evaluate_model(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.DataFrame, threshold: float):
    print("[evaluation] - INFO: Evaluating model performance on test data...")
    # Models supported by scikit-learn may either be probabilistic-based or not, check is required
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = [1 if score >= threshold else 0 for score in y_pred_proba]
    else:
        y_pred = model.predict(X_test)

    # Print results
    print('='*80)
    print("Accuracy score:", round(accuracy_score(y_test, y_pred), 4))
    print("F1 score:", round(f1_score(y_test, y_pred), 4))
    print("ROC AUC score:", round(roc_auc_score(y_test, y_pred), 4))
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    print("PR AUC score:", round(auc(recall, precision), 4))
    print(classification_report(y_test, y_pred, labels=[0, 1]))
    print('='*80)
    return

def load_model(model_file_path: str):
    model = joblib.load(model_file_path)
    return model

def read_data(x_test_data_path: str, y_test_data_path: str):
    X_test = pd.read_csv(x_test_data_path, sep=',')
    y_test = pd.read_csv(y_test_data_path, sep=',')

    return X_test, y_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model performance using test data.')
    parser.add_argument('--model_file_path', type=str, help='Path to the joblib model file.')
    parser.add_argument('--x_test_data_path', type=str, help='Path to the test reads CSV.')
    parser.add_argument('--y_test_data_path', type=str, help='Path to the test data labels.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probabilistic threshold for binary classification')
    args = parser.parse_args()

    model = load_model(args.model_file_path)
    X_test, y_test = read_data(args.x_test_data_path, args.y_test_data_path)
    evaluate_model(model, X_test, y_test, args.threshold)