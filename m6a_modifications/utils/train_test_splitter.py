"""
Data Splitting Module

This module provides a function to split a merged dataset into training
and testing sets based on unique gene identifiers. The splitting is
done randomly while maintaining the distribution of unique genes.

Required Libraries:
- random: For shuffling the unique genes.
- pandas: For data manipulation and analysis.

Functions:
- split_data: Splits the merged dataset into training and testing sets.

Usage:
    This module is intended to be imported and its functions called
    within a data processing pipeline where data needs to be split
    into training and testing subsets.
"""

import random

import pandas as pd


def split_data(
    merged_data: pd.DataFrame, train_data_ratio: float = 0.8, seed: int = 888
):
    """
    Split the merged dataset into training and testing sets based on unique gene identifiers.

    This function retrieves unique genes from the merged dataset, shuffles them,
    and splits them into training and testing sets according to the specified
    training data ratio.

    Args:
        merged_data (pd.DataFrame): The dataset to be split, containing a 'gene_id' column.
        train_data_ratio (float): The ratio of data to be used for training (default is 0.8).
        seed (int): Random seed for reproducibility (default is 888).

    Returns:
        tuple: A tuple containing two DataFrames: the training data and the testing data.
    """

    # Set seed for randomness
    random.seed(seed)

    # Retrieve all unique genes
    unique_genes = set()
    for data_row in merged_data.iloc:
        unique_genes.add(data_row['gene_id'])

    # Shuffle genes to get random permutation
    unique_genes = sorted(list(unique_genes)) # The unique genes are sorted as the conversion from set to list may result in a random permutation that cannot be reproduced
    random.shuffle(unique_genes)

    # Split the genes into training and testing sets
    train_genes, test_genes = (
        unique_genes[: int(len(unique_genes) * train_data_ratio)],
        unique_genes[int(len(unique_genes) * train_data_ratio) :],
    )

    # Create the training and testing dataframes
    train_data = merged_data[merged_data['gene_id'].isin(train_genes)]
    test_data = merged_data[merged_data['gene_id'].isin(test_genes)]

    return train_data, test_data
