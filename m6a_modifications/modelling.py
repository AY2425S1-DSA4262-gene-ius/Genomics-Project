"""
Script for training a Histogram-based Gradient Boosting model.

This script loads training data, trains a Histogram-based Gradient Boosting model using
the scikit-learn library, and saves the trained model as a joblib file.

Modules Required:
- argparse: For parsing command-line arguments.
- os: For directory and file handling.
- joblib: For saving the trained model.
- pandas: For data manipulation and handling CSV files.
- sklearn.ensemble: For the Histogram-based Gradient Boosting Classifier.

Usage:
    python -m m6a_modifications.modelling --x_train_data_path <path> --y_train_data_path <path> --seed <seed>

Example:
    python -m m6a_modifications.modelling --x_train_data_path X_train.csv --y_train_data_path y_train.csv --seed 42
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

MODEL = 'Histogram-based_Gradient_Boosting'

def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, seed: int):
    """
    Train a Histogram-based Gradient Boosting model on the provided training data.

    Args:
        X_train (pd.DataFrame): Features of the training dataset.
        y_train (pd.DataFrame): Labels corresponding to the training dataset.
        seed (int): Random seed for reproducibility.

    Returns:
        HistGradientBoostingClassifier: The trained model.
    """

    print(f'[modelling] - INFO: Initialising model [{MODEL}]...')

    # Training!
    model = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100, max_leaf_nodes=31, min_samples_leaf=20, random_state=seed)
    model.fit(X_train, y_train.values.ravel())
    print(f'[modelling] - INFO: Model [{MODEL}] has been trained successfully')

    # Save the model as a joblib file
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{MODEL}.joblib')
    print(f'[modelling] - INFO: Model has been saved: models/{MODEL}.joblib')

    return model


def read_data(x_train_data_path: str, y_train_data_path: str):
    """
    Read training features and labels from CSV files.

    Args:
        x_train_data_path (str): Path to the CSV file containing training features.
        y_train_data_path (str): Path to the CSV file containing training labels.

    Returns:
        tuple: DataFrames of features (X_train) and labels (y_train).
    """

    X_train = pd.read_csv(x_train_data_path, sep=',')
    y_train = pd.read_csv(y_train_data_path, sep=',')

    return X_train, y_train


if __name__ == '__main__':
    # Set up argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description='Train the model using train data.')

    # Define command-line arguments
    parser.add_argument(
        '--x_train_data_path', type=str, help='Path to the train reads CSV.'
    )
    parser.add_argument(
        '--y_train_data_path', type=str, help='Path to the train data labels.'
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='Seed for reproducibility.'
    )

    # Parse arguments
    args = parser.parse_args()

    # Read training data
    X_train, y_train = read_data(args.x_train_data_path, args.y_train_data_path)

    # Train the model
    train_model(X_train, y_train, args.seed)
