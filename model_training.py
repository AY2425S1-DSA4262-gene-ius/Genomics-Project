"""
Script to execute the full training pipeline for m6a modifications.

This script loads data, preprocesses it, splits it into training and testing sets, trains a model, 
and evaluates it on the test set. The main function, `start`, orchestrates the workflow, 
and argparse is used to handle command-line arguments.

Modules Required:
- argparse: For parsing command-line arguments
- pandas: For handling data in DataFrame format
- m6a_modifications.data_processing.process_and_split_data: Processes and splits data into train/test sets
- m6a_modifications.evaluation.evaluate_model: Evaluates the model
- m6a_modifications.modelling.train_model: Trains the model
- m6a_modifications.raw_data_preparer.prepare_data: Prepares raw data for processing

Usage:
    python -m model_training --data_file_path <path> --labels_data_path <path> --train_data_ratio <ratio>
                     --threshold <threshold> --output_file_name <filename> --seed <seed>

Example:
    python -m model_training --data_file_path data.json.gz --labels_data_path labels.csv --train_data_ratio 0.8 
                     --threshold 0.5 --output_file_name results.csv --seed 42
"""

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
    """
    Main function to execute the data preparation, processing, training, and evaluation pipeline.

    Args:
        data_file_path (str): Path to the gzipped dataset JSON file containing read data.
        labels_data_path (str): Path to the CSV file containing labels for the dataset.
        train_data_ratio (float): Ratio of data to use for training (e.g., 0.8 for 80%).
        threshold (float): Probability threshold for binary classification decisions.
        output_file_name (str): Filename to save the evaluation results.
        seed (int): Random seed for reproducibility.
    """
    # Step 1: Load and prepare the raw read data using `prepare_data`.
    reads_data = prepare_data(data_file_path)

    # Step 2: Load labels data from a CSV file to align with the reads data.
    labels_data = pd.read_csv(labels_data_path, sep=',')

    # Step 3: Split data into train and test sets, including any necessary preprocessing.
    X_train, y_train, X_test, y_test, X_train_identity, X_test_identity = process_and_split_data(
        reads_data, labels_data, train_data_ratio, seed
    )

    # Step 4: Train the model on the training dataset with a given random seed for reproducibility.
    model = train_model(X_train, y_train, seed)

    # Step 5: Evaluate the model using test data and specified probabilistic threshold for classification.
    evaluate_model(model, X_test, y_test, X_test_identity, output_file_name, threshold)
    return


if __name__ == '__main__':
    # Argument parser setup for command-line execution
    parser = argparse.ArgumentParser(
        description='Execute the full ML training pipeline.'
    )

    # Argument definitions for command-line inputs
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

    # Parsing command-line arguments
    args = parser.parse_args()

    # Initiate the full training pipeline using the parsed arguments
    start(
        args.data_file_path,
        args.labels_data_path,
        args.train_data_ratio,
        args.threshold,
        args.output_file_name,
        args.seed,
    )
