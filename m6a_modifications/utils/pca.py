"""
Principal Component Analysis (PCA) for Dimensionality Reduction

This module provides functions to perform Principal Component Analysis (PCA)
on datasets. It includes methods for fitting PCA on training data, transforming
test data, and transforming new datasets using a pre-trained PCA model.

Required Libraries:
- joblib: For loading pre-trained PCA models.
- pandas: For data manipulation and analysis.
- sklearn.decomposition.PCA: For performing PCA.

Functions:
- run_split_pca: Fits PCA on training data and transforms both training and test data.
- run_pca: Transforms a given dataset using a pre-trained PCA model.

Usage:
    This module is intended to be imported and its functions called
    within a data processing pipeline.
"""

import joblib
import pandas as pd
from sklearn.decomposition import PCA


def run_split_pca(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit PCA on the training data and transform both training and test data.

    This function applies PCA to reduce the dimensionality of the datasets
    while preserving 95% of the variance.

    Args:
        X_train (pd.DataFrame): The training dataset with features.
        X_test (pd.DataFrame): The testing dataset with features.

    Returns:
        tuple: A tuple containing the PCA-transformed training data, 
        PCA-transformed testing data, and the fitted PCA model.
    """

    # Initialise PCA and apply with 95% variance explained
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Create new column names for the principal components
    new_column_names = [f'PC_{i+1}' for i in range(X_train_pca.shape[1])]

    # Create DataFrames for the PCA-transformed data
    X_train = pd.DataFrame(X_train_pca, columns=new_column_names)
    X_test = pd.DataFrame(X_test_pca, columns=new_column_names)

    # Print information about the selected components
    final_components = pca.n_components_
    exp_var_ratio = pca.explained_variance_ratio_
    print(
        f'[run_pca] - INFO: Using {final_components} principal components, we can explain {exp_var_ratio.sum() * 100:.2f}% of the variance in the original data'
    )

    return X_train, X_test, pca

def run_pca(data: pd.DataFrame, pca_path: str):
    """
    Transform a dataset using a pre-trained PCA model.

    This function uses a previously fitted PCA model to transform the provided
    dataset, reducing its dimensionality.

    Args:
        data (pd.DataFrame): The dataset with features to be transformed.
        pca_path (str): Path to the pre-trained PCA model.

    Returns:
        pd.DataFrame: The PCA-transformed dataset.
    """

    # Initialise PCA and apply with 95% variance explained
    pca = joblib.load(pca_path)
    pca_data = pca.transform(data)

    # Create new column names for the principal components
    new_column_names = [f'PC_{i+1}' for i in range(pca_data.shape[1])]

    # Create DataFrames for the PCA-transformed data
    data = pd.DataFrame(pca_data, columns=new_column_names)

    # Print information about the selected components
    final_components = pca.n_components_
    exp_var_ratio = pca.explained_variance_ratio_
    print(
        f'[run_pca] - INFO: Using {final_components} principal components, we can explain {exp_var_ratio.sum() * 100:.2f}% of the variance in the original data'
    )

    return data
