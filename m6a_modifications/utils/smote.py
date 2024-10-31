"""
Synthetic Minority Over-sampling Technique (SMOTE) for Imbalanced Datasets

This module provides a function to apply the SMOTE algorithm for
oversampling minority classes in imbalanced datasets. SMOTE generates
synthetic samples to balance the class distribution.

Required Libraries:
- pandas: For data manipulation and analysis.
- imblearn.over_sampling.SMOTE: For applying the SMOTE algorithm.

Functions:
- run_smote: Applies SMOTE to the training dataset to generate synthetic samples.

Usage:
    This module is intended to be imported and its functions called
    within a data processing pipeline for handling imbalanced datasets.
"""

import pandas as pd
from imblearn.over_sampling import SMOTE


def run_smote(X_train: pd.DataFrame, y_train: pd.DataFrame, seed: int = 42):
    """
    Apply SMOTE to oversample the minority class in the training dataset.

    This function uses the SMOTE algorithm to generate synthetic samples
    for the minority class, thus helping to balance the class distribution.

    Args:
        X_train (pd.DataFrame): The training dataset with features.
        y_train (pd.DataFrame): The corresponding labels for the training dataset.
        seed (int): Random seed for reproducibility (default is 42).

    Returns:
        tuple: A tuple containing the oversampled feature dataset (X_train)
        and the corresponding labels (y_train).
    """

    # Initialise SMOTE and oversample
    smote = SMOTE(random_state=seed)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, y_train
