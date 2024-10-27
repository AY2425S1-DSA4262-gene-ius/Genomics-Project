import argparse
import os
from typing import Optional

import joblib
import pandas as pd

from m6a_modifications.utils.data_transform import (
    engineer_features,
    standardise_data,
    standardise_split_data,
)
from m6a_modifications.utils.pca import run_pca, run_split_pca
from m6a_modifications.utils.smote import run_smote
from m6a_modifications.utils.train_test_splitter import split_data


def process_and_split_data(
    reads_data: pd.DataFrame,
    labels_data: pd.DataFrame,
    train_data_ratio: float,
    seed: int,
):
    print('[data_processing] - INFO: Executing feature engineering...')
    # Feature engineer the reads data
    engineered_reads_data = engineer_features(reads_data)

    # Data type inference by pandas may cause merge error between tables. Standardising the data type of merged columns.
    engineered_reads_data.transcript_position = (
        engineered_reads_data.transcript_position.astype(int)
    )
    labels_data.transcript_position = labels_data.transcript_position.astype(int)
    
    # Combine the reads data and labels data into a single dataframe
    merged_data = pd.merge(
        engineered_reads_data,
        labels_data,
        on=['transcript_id', 'transcript_position'],
        how='inner',
    )

    # Miscellaneous - Reorder columns
    cols = ['gene_id'] + [col for col in merged_data.columns if col != 'gene_id']
    merged_data = merged_data[cols]

    # Train-test split
    print(
        f'[data_processing] - INFO: Splitting the data into train and test with the ratio of {train_data_ratio * 100}:{(1 - train_data_ratio) * 100}...'
    )
    train_data, test_data = split_data(merged_data, train_data_ratio, seed)

    # Standardisation of numerical features
    print('[data_processing] - INFO: Standardising numerical features...')
    train_data, test_data, scaler = standardise_split_data(train_data, test_data)

    # Save the StandardScaler as a joblib file under `artifacts`
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(scaler, 'artifacts/standard_scaler.joblib')
    print(
        '[data_processing] - INFO: Saved the fitted StandardScaler as joblib: .\\artifacts\\standard_scaler.joblib'
    )

    # Retain X and y for training
    X_train_identity = train_data[
        ['gene_id', 'transcript_id', 'transcript_position', 'Sequence']
    ]
    X_test_identity = test_data[
        ['gene_id', 'transcript_id', 'transcript_position', 'Sequence']
    ]
    X_train = train_data.drop(
        ['gene_id', 'transcript_id', 'transcript_position', 'Sequence', 'label'], axis=1
    )
    X_test = test_data.drop(
        ['gene_id', 'transcript_id', 'transcript_position', 'Sequence', 'label'], axis=1
    )
    y_train = train_data['label']
    y_test = test_data['label']

    # Execute Principal Component Analysis to reduce features
    print(
        '[data_processing] - INFO: Running Principal Component Analysis on the numerical features...'
    )
    X_train, X_test, pca = run_split_pca(X_train, X_test)

    # Save the PCA artifact as a joblib file under `artifacts`
    joblib.dump(pca, 'artifacts/pca.joblib')
    print(
        '[data_processing] - INFO: Saved the fitted PCA as joblib: .\\artifacts\\pca.joblib'
    )

    # Balance the imbalanced dataset using SMOTE
    print('[data_processing] - INFO: Excuting oversampling using SMOTE...')
    y_train_zeros_count = y_train.value_counts()[0]
    y_train_ones_count = y_train.value_counts()[1]
    X_train, y_train = run_smote(X_train, y_train)
    print(
        f'[data_processing] - DEBUG: Training data has been oversampled [0s: {y_train_zeros_count}, 1s: {y_train_ones_count} ---> 0s: {y_train.value_counts()[0]}, 1s: {y_train.value_counts()[1]}]'
    )

    # Save the outputs as a CSV file
    X_train.to_csv('data/X_train.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    print(
        '[data_processing] - INFO: Train and Test data have been saved: data/X_train.csv | data/y_train.csv | data/X_test.csv | data/y_test.csv'
    )

    X_train_identity.to_csv('data/X_train_identity.csv', index=False)
    X_test_identity.to_csv('data/X_test_identity.csv', index=False)
    print(
        '[data_processing] - INFO: Train and Test identity data have been saved: data/X_train_identity.csv | data/X_test_identity.csv'
    )

    return X_train, y_train, X_test, y_test, X_train_identity, X_test_identity
    

def process_data(
    reads_data: pd.DataFrame,
    standard_scaler_path: str,
    pca_path: str,
):
    print('[data_processing] - INFO: Executing feature engineering...')
    # Feature engineer the reads data
    engineered_reads_data = engineer_features(reads_data)

    # Standardisation of numerical features
    print('[data_processing] - INFO: Standardising numerical features...')
    scaled_data = standardise_data(engineered_reads_data, standard_scaler_path)

    # Retain identity for evaluation
    data_identity = scaled_data[
        ['transcript_id', 'transcript_position', 'Sequence']
    ]
    data_features = scaled_data.drop(
        ['transcript_id', 'transcript_position', 'Sequence'], axis=1
    )

    # Execute Principal Component Analysis to reduce features
    print(
        '[data_processing] - INFO: Running Principal Component Analysis on the numerical features...'
    )
    pca_data = run_pca(data_features, pca_path)

    # Save the outputs as a CSV file
    pca_data.to_csv('data/processed_data.csv', index=False)
    print(
        '[data_processing] - INFO: Data have been saved: data/processed_data.csv'
    )

    data_identity.to_csv('data/processed_data_identity.csv', index=False)
    print(
        '[data_processing] - INFO: Identity data been saved: data/processed_data_identity.csv'
    )

    return pca_data, data_identity

def read_data(reads_data_path: str, labels_data_path: str):
    print(
        f'[data_processing] - INFO: Reading data and its labels from: {reads_data_path} | {labels_data_path}'
    )
    reads_data = pd.read_csv(reads_data_path, sep=',')
    labels_data = pd.read_csv(labels_data_path, sep=',')

    return reads_data, labels_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process data, of which includes feature engineering and cleaning up.'
    )
    parser.add_argument(
        '--reads_data_path', type=str, help='Path to the prepared reads CSV.'
    )
    parser.add_argument('--labels_data_path', type=str, help='Path to the data labels.')
    parser.add_argument(
        '--train_data_ratio',
        type=float,
        default=0.8,
        help='Ratio for train data in train test split.',
    )
    parser.add_argument('--standard_scaler_path', type=str, help='Path to the fitted StandardScaler.')
    parser.add_argument('--pca_path', type=str, help='Path to the fitted PCA artifact.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Seed for reproducibility.'
    )
    args = parser.parse_args()

    if args.train_data_ratio:
        print(
            f'[data_processing] - INFO: Reading data and its labels from: {args.reads_data_path} | {args.labels_data_path}'
        )
        reads_data = pd.read_csv(args.reads_data_path, sep=',')
        labels_data = pd.read_csv(args.labels_data_path, sep=',')
        process_and_split_data(reads_data, labels_data, args.train_data_ratio, args.seed)
    else:
        print(
            f'[data_processing] - INFO: Reading data from: {args.reads_data_path}'
        )
        reads_data = pd.read_csv(args.reads_data_path, sep=',')
        process_data(reads_data, args.standard_scaler_path, args.pca_path)
