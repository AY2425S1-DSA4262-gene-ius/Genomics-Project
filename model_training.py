import argparse

import pandas as pd

from m6a_modifications.data_processing import process_and_split_data
from m6a_modifications.evaluation import evaluate_model
from m6a_modifications.modelling import train_model
from m6a_modifications.raw_data_preparer import prepare_data


def start(
    data_file_path: str,
    labels_data_path: str,
    train_data_ratio: float,
    threshold: float,
    output_file_name: str,
    seed: int,
):
    # Prepare the reads data and the labels data
    reads_data = prepare_data(data_file_path)
    labels_data = pd.read_csv(labels_data_path, sep=',')

    # Prepare the train and test data, of which includes data preprocessing and feature engineering
    X_train, y_train, X_test, y_test, X_train_identity, X_test_identity = process_and_split_data(
        reads_data, labels_data, train_data_ratio, seed
    )

    # Train the model and evaluate
    model = train_model(X_train, y_train, seed)
    evaluate_model(model, X_test, y_test, X_test_identity, output_file_name, threshold)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Execute the full ML training pipeline.'
    )
    parser.add_argument(
        '--data_file_path', type=str, help='Path to the gzipped dataset json file.'
    )
    parser.add_argument('--labels_data_path', type=str, help='Path to the labels file.')
    parser.add_argument(
        '--train_data_ratio',
        type=float,
        default=0.8,
        help='Ratio for train data in train test split.',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Probabilistic threshold for binary classification',
    )
    parser.add_argument(
        '--output_file_name',
        type=str,
        help='Filename of output',
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='Seed for reproducibility.'
    )

    args = parser.parse_args()
    start(
        args.data_file_path,
        args.labels_data_path,
        args.train_data_ratio,
        args.threshold,
        args.output_file_name,
        args.seed,
    )
