import random

import pandas as pd


def split_data(
    merged_data: pd.DataFrame, train_data_ratio: float = 0.8, seed: int = 42
):
    # Set seed for randomness
    random.seed(seed)

    # Retrieve all unique genes
    unique_genes = set()
    for data_row in merged_data.iloc:
        unique_genes.add(data_row['gene_id'])

    # Shuffle genes to get random permutation
    unique_genes = list(unique_genes)
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
