"""
Script for processing, feature engineering, and splitting of data.

This script performs data preparation tasks, including feature engineering, standardisation, 
Principal Component Analysis (PCA), and train-test splitting with optional SMOTE oversampling 
to address data imbalance. It can save processed data and artifacts for later model training 
and evaluation.

Modules Required:
- argparse: For command-line argument parsing
- os: For directory and file handling
- joblib: For saving trained scalers and PCA models
- pandas: For handling data in DataFrame format
- m6a_modifications.utils: Contains data transformation and utility functions for feature engineering, PCA, SMOTE, etc.

Usage:
    python -m m6a_modifications.data_processing --reads_data_path <path> --labels_data_path <path> --train_data_ratio <ratio>
                     --standard_scaler_path <path> --pca_path <path> --seed <seed>

Example:
    python -m m6a_modifications.data_processing --reads_data_path reads.csv --labels_data_path labels.csv --train_data_ratio 0.8 
                     --standard_scaler_path artifacts/standard_scaler.joblib --pca_path artifacts/pca.joblib 
                     --seed 888
"""

import argparse
import os

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
    """
    Process and split data for training and evaluation.

    This function performs feature engineering, standardisation, PCA, train-test splitting, 
    and SMOTE oversampling to balance the dataset. The processed data and metadata 
    are saved as CSV files for future use.

    Args:
        reads_data (pd.DataFrame): DataFrame with read data.
        labels_data (pd.DataFrame): DataFrame with label data.
        train_data_ratio (float): Ratio of data to use for training.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Processed train/test datasets and their metadata.
    """

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
        f'[data_processing] - INFO: Splitting the data into train and test with the ratio of {round(train_data_ratio * 100)}:{round((1 - train_data_ratio) * 100)}...'
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
    X_train, X_test, pca = run_split_pca(X_train, X_test, seed)

    # Save the PCA artifact as a joblib file under `artifacts`
    joblib.dump(pca, 'artifacts/pca.joblib')
    print(
        '[data_processing] - INFO: Saved the fitted PCA as joblib: .\\artifacts\\pca.joblib'
    )

    # Balance the imbalanced dataset using SMOTE
    print('[data_processing] - INFO: Excuting oversampling using SMOTE...')
    y_train_zeros_count = y_train.value_counts()[0]
    y_train_ones_count = y_train.value_counts()[1]
    X_train, y_train = run_smote(X_train, y_train, seed)
    print(
        f'[data_processing] - DEBUG: Training data has been oversampled [0s: {y_train_zeros_count}, 1s: {y_train_ones_count} ---> 0s: {y_train.value_counts()[0]}, 1s: {y_train.value_counts()[1]}]'
    )

    # Save the outputs as a CSV file
    os.makedirs('processed_data', exist_ok=True)
    X_train.to_csv('processed_data/X_train.csv', index=False)
    y_train.to_csv('processed_data/y_train.csv', index=False)
    X_test.to_csv('processed_data/X_test.csv', index=False)
    y_test.to_csv('processed_data/y_test.csv', index=False)
    print(
        '[data_processing] - INFO: Train and Test data have been saved: processed_data/X_train.csv | processed_data/y_train.csv | processed_data/X_test.csv | processed_data/y_test.csv'
    )

    X_train_identity.to_csv('processed_data/X_train_identity.csv', index=False)
    X_test_identity.to_csv('processed_data/X_test_identity.csv', index=False)
    print(
        '[data_processing] - INFO: Train and Test identity data have been saved: processed_data/X_train_identity.csv | processed_data/X_test_identity.csv'
    )

    return X_train, y_train, X_test, y_test, X_train_identity, X_test_identity
    

def process_data(
    reads_data: pd.DataFrame,
    standard_scaler_path: str,
    pca_path: str,
):
    """
    Process data without splitting for prediction or inference use.

    Args:
        reads_data (pd.DataFrame): DataFrame with read data to process.
        standard_scaler_path (str): Path to a saved StandardScaler model.
        pca_path (str): Path to a saved PCA model.

    Returns:
        tuple: Processed feature data and identity metadata for evaluation.
    """

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
    os.makedirs('processed_data', exist_ok=True)
    pca_data.to_csv('processed_data/processed_data.csv', index=False)
    print(
        '[data_processing] - INFO: Data have been saved: processed_data/processed_data.csv'
    )

    data_identity.to_csv('processed_data/processed_data_identity.csv', index=False)
    print(
        '[data_processing] - INFO: Identity data been saved: processed_data/processed_data_identity.csv'
    )

    return pca_data, data_identity

def read_data(reads_data_path: str, labels_data_path: str):
    """
    Read the read data and labels data from CSV files.

    Args:
        reads_data_path (str): Path to the read data CSV.
        labels_data_path (str): Path to the labels data CSV.

    Returns:
        tuple: Read data and labels DataFrames.
    """

    print(
        f'[data_processing] - INFO: Reading data and its labels from: {reads_data_path} | {labels_data_path}'
    )
    reads_data = pd.read_csv(reads_data_path, sep=',')
    labels_data = pd.read_csv(labels_data_path, sep=',')

    return reads_data, labels_data


if __name__ == '__main__':
    # Set up argparse to parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Process data, of which includes feature engineering and cleaning up.'
    )

    # Define command-line arguments
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
        '--seed', type=int, default=888, help='Seed for reproducibility.'
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute the appropriate processing function based on arguments
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
