import argparse
import os
from typing import Optional

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def evaluate_model(
    model: BaseEstimator, data: pd.DataFrame, labels: Optional[pd.DataFrame], data_identity: pd.DataFrame, output_file_name: str, threshold: float = 0.5
):
    print('[evaluation] - INFO: Running model prediction...')
    # Models supported by scikit-learn may either be probabilistic-based or not, a check is required
    if hasattr(model, 'predict_proba'):
        y_pred = model.predict_proba(data)[:, 1]
        y_pred_binary = [1 if score >= threshold else 0 for score in y_pred]
    else:
        y_pred = model.predict(data)
        y_pred_binary = y_pred

    # Generate prediction output table
    selected_data_identity = data_identity[['transcript_id', 'transcript_position']].reset_index(drop=True)
    results = pd.concat(
        [selected_data_identity, pd.DataFrame({'score': y_pred})],
        axis=1
    )

    os.makedirs('predictions', exist_ok=True)
    results.to_csv(f'predictions/{output_file_name}', index=False)
    print(f'[evaluation] - INFO: Saved evaluation as a CSV: predictions/{output_file_name}')
    
    if labels is not None:
        print('[evaluation] - INFO: Evaluating model performance on test data...')

        # Print results
        print('=' * 80)
        print('Accuracy score:', round(accuracy_score(labels, y_pred_binary), 4))
        print('F1 score:', round(f1_score(labels, y_pred_binary), 4))
        print('ROC AUC score:', round(roc_auc_score(labels, y_pred_binary), 4))
        precision, recall, thresholds = precision_recall_curve(labels, y_pred_binary)
        print('PR AUC score:', round(auc(recall, precision), 4))
        print(classification_report(labels, y_pred_binary, labels=[0, 1]))
        print('=' * 80)
        return


def load_model(model_file_path: str):
    model = joblib.load(model_file_path)
    return model


def read_data(data_path: str, labels_path: str, data_identity_path: str):
    data = pd.read_csv(data_path, sep=',')
    labels = pd.read_csv(labels_path, sep=',')
    data_identity = pd.read_csv(data_identity_path, sep=',')

    return data, labels, data_identity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate model performance using test data.'
    )
    parser.add_argument(
        '--model_file_path', type=str, help='Path to the joblib model file.'
    )
    parser.add_argument(
        '--data_path', type=str, help='Path to the reads CSV.'
    )
    parser.add_argument(
        '--data_identity_path', type=str, help='Identity of each data in the reads CSV.'
    )
    parser.add_argument(
        '--labels_path', type=str, required=False, help='Path to the labels data.'
    )
    parser.add_argument(
        '--output_file_name',
        type=str,
        help='Filename of output',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Probabilistic threshold for binary classification',
    )
    args = parser.parse_args()

    model = load_model(args.model_file_path)
    data, labels, data_identity = read_data(args.data_path, args.labels_path, args.data_identity_path)
    evaluate_model(model, data, labels, data_identity, args.output_file_name, args.threshold)
