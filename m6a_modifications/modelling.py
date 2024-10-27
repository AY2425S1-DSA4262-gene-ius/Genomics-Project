import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

MODEL = 'Histogram-based_Gradient_Boosting'

def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, seed: int):
    print(f'[modelling] - INFO: Initialising model [{MODEL}]...')

    # Training!
    model = HistGradientBoostingClassifier(learning_rate=0.2, max_depth=20, max_iter=300, max_leaf_nodes=127, min_samples_leaf=20, random_state=seed)
    model.fit(X_train, y_train.values.ravel())
    print(f'[modelling] - INFO: Model [{MODEL}] has been trained successfully')

    # Save the model as a joblib file
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{MODEL}.joblib')
    print(f'[modelling] - INFO: Model has been saved: models/{MODEL}.joblib')

    return model


def read_data(x_train_data_path: str, y_train_data_path: str):
    X_train = pd.read_csv(x_train_data_path, sep=',')
    y_train = pd.read_csv(y_train_data_path, sep=',')

    return X_train, y_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model using train data.')
    parser.add_argument(
        '--x_train_data_path', type=str, help='Path to the train reads CSV.'
    )
    parser.add_argument(
        '--y_train_data_path', type=str, help='Path to the train data labels.'
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='Seed for reproducibility.'
    )
    args = parser.parse_args()

    X_train, y_train = read_data(args.x_train_data_path, args.y_train_data_path)
    train_model(X_train, y_train, args.seed)
