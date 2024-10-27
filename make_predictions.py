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
    # Prepare the reads data
    reads_data = prepare_data(data_file_path)

    # Prepare the train and test data, of which includes data preprocessing and feature engineering
    data, data_identity = process_data(
        reads_data, standard_scaler_path, pca_path
    )

    # Load the model
    model = joblib.load(model_path)

    evaluate_model(model=model, data=data, data_identity=data_identity, output_file_name=output_file_name, labels=None)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Execute prediction on dataset.'
    )
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

    args = parser.parse_args()
    start(
        args.data_file_path,
        args.model_path,
        args.standard_scaler_path,
        args.pca_path,
        args.output_file_name,
    )
