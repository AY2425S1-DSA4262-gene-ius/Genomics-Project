"""
Script to execute prediction on a dataset for m6a modifications.

This script loads prepared read data, applies preprocessing and feature engineering,
and evaluates a trained model on the dataset. The main function, `start`, initialises
and executes the workflow, while argparse is used to handle command-line arguments.

Modules Required:
- argparse: For parsing command-line arguments
- joblib: For loading trained model and preprocessing artifacts
- m6a_modifications.data_processing.process_data: Processes input data
- m6a_modifications.evaluation.evaluate_model: Evaluates the model
- m6a_modifications.raw_data_preparer.prepare_data: Prepares raw data for processing

Usage:
    python -m make_predictions --data_file_path <path> --model_path <path> --standard_scaler_path <path> 
                     --pca_path <path> --output_file_name <filename>

Example:
    python -m make_predictions --data_file_path data.json.gz --model_path model.joblib --standard_scaler_path scaler.joblib 
                     --pca_path pca.joblib --output_file_name results.csv
"""

import argparse

import joblib

from m6a_modifications.data_processing import process_data
from m6a_modifications.evaluation import evaluate_model
from m6a_modifications.raw_data_preparer import prepare_data


def start(
    data_file_path: str,
    model_path: str,
    standard_scaler_path: str,
    pca_path: str,
    output_file_name: str,
):
    """
    Main function to execute the data preparation, processing, and evaluation pipeline.

    Args:
        data_file_path (str): Path to the gzipped dataset JSON file.
        model_path (str): Path to the trained model file (in .joblib format).
        standard_scaler_path (str): Path to the fitted StandardScaler object (in .joblib format).
        pca_path (str): Path to the fitted PCA object (in .joblib format).
        output_file_name (str): Filename to save the evaluation results.
    """
    # Step 1: Prepare the raw read data using `prepare_data`.
    reads_data = prepare_data(data_file_path)

    # Step 2: Process the data to prepare train and test sets, 
    # including necessary preprocessing steps like scaling and PCA transformation.
    data, data_identity = process_data(
        reads_data, standard_scaler_path, pca_path
    )

    # Step 3: Load the pretrained model.
    model = joblib.load(model_path)

    # Step 4: Evaluate the model on the processed data.
    evaluate_model(model=model, data=data, data_identity=data_identity, output_file_name=output_file_name, labels=None)
    return


if __name__ == '__main__':
    # Argument parser setup for command-line execution
    parser = argparse.ArgumentParser(
        description='Execute prediction on dataset.'
    )

    # Argument definitions for the script
    parser.add_argument(
        '--data_file_path', type=str, help='Path to the gzipped dataset json file.'
    )
    parser.add_argument('--model_path', type=str, help='Path to the trained model.')
    parser.add_argument('--standard_scaler_path', type=str, help='Path to the fitted StandardScaler.')
    parser.add_argument('--pca_path', type=str, help='Path to the fitted PCA artifact.')
    parser.add_argument(
        '--output_file_name',
        type=str,
        help='Filename of output',
    )

    # Parsing arguments from command-line input
    args = parser.parse_args()

    # Starting the prediction and evaluation workflow using provided arguments
    start(
        args.data_file_path,
        args.model_path,
        args.standard_scaler_path,
        args.pca_path,
        args.output_file_name,
    )
