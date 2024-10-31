"""
Feature Engineering and Standardisation for Transcript Data

This module provides functions to engineer features from transcript data,
including calculating differences, aggregating features, extracting motifs,
and standardising datasets using z-score normalisation.

Required Libraries:
- joblib: For loading pre-trained scalers.
- pandas: For data manipulation and analysis.
- sklearn.preprocessing.StandardScaler: For standardising feature data.

Functions:
- engineer_features: Creates new features from the raw transcript data.
- standardise_split_data: Standardises training and testing datasets.
- standardise_data: Standardises a given dataset using a pre-trained scaler.

Usage:
    This module is intended to be imported and its functions called
    within a data processing pipeline.
"""

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None

# Possible middle 5-mers
dra_ch_motifs = [
    'AAACA', 'AAACC', 'AAACT', 'AGACA', 'AGACC', 'AGACT',
    'GAACA', 'GAACC', 'GAACT', 'GGACA', 'GGACC', 'GGACT',
    'TAACA', 'TAACC', 'TAACT', 'TGACA', 'TGACC', 'TGACT',
]

# Possible front 5-mers
pos_1_5 = [
    'GAAAC', 'AAAAC', 'CAAAC', 'TAAAC', 'GAGAC', 'AAGAC',
    'CAGAC', 'TAGAC', 'GGAAC', 'AGAAC', 'CGAAC', 'TGAAC',
    'GGGAC', 'AGGAC', 'CGGAC', 'TGGAC', 'GTAAC', 'ATAAC',
    'CTAAC', 'TTAAC', 'GTGAC', 'ATGAC', 'CTGAC', 'TTGAC',
]

# Possible back 5-mers
pos_3_7 = [
    'AACAG', 'AACAA', 'AACAC', 'AACAT', 'AACCG', 'AACCA',
    'AACCC', 'AACCT', 'AACTG', 'AACTA', 'AACTC', 'AACTT',
    'GACAG', 'GACAA', 'GACAC', 'GACAT', 'GACCG', 'GACCA',
    'GACCC', 'GACCT', 'GACTG', 'GACTA', 'GACTC', 'GACTT',
]


def engineer_features(reads_data: pd.DataFrame):
    """
    Engineer new features from the raw transcript data.

    This function calculates differences between dwelling times and 
    statistics (mean, median, standard deviation) of signals, aggregates 
    data by transcript ID and position, and extracts 5-mer sequences.

    Args:
        reads_data (pd.DataFrame): Raw transcript data with dwelling times 
        and other signal measurements.

    Returns:
        pd.DataFrame: A DataFrame with engineered features.
    """

    # New Features - differences between dwelling times, mean and SD of signals.
    columns_to_compare = ['Dwelling_Time', 'Mean', 'SD']
    for col in columns_to_compare:
        reads_data[f'Difference_{col}_1_2'] = (
            reads_data[f'{col}_2'] - reads_data[f'{col}_1']
        )
        reads_data[f'Difference_{col}_2_3'] = (
            reads_data[f'{col}_2'] - reads_data[f'{col}_3']
        )

    # Aggregate by transcript_id, transcript_position, and Sequence
    agg_funcs = ['min', 'max', 'mean', 'median', 'std']
    reads_data_grouped = reads_data.groupby(
        ['transcript_id', 'transcript_position', 'Sequence']
    ).agg(
        {
            'Dwelling_Time_1': agg_funcs,
            'SD_1': agg_funcs,
            'Mean_1': agg_funcs,
            'Dwelling_Time_2': agg_funcs,
            'SD_2': agg_funcs,
            'Mean_2': agg_funcs,
            'Dwelling_Time_3': agg_funcs,
            'SD_3': agg_funcs,
            'Mean_3': agg_funcs,
            'Difference_Dwelling_Time_1_2': agg_funcs,
            'Difference_Dwelling_Time_2_3': agg_funcs,
            'Difference_Mean_1_2': agg_funcs,
            'Difference_Mean_2_3': agg_funcs,
            'Difference_SD_1_2': agg_funcs,
            'Difference_SD_2_3': agg_funcs,
        }
    ).fillna(0)

    # Reset the index to bring 'transcript_id', 'transcript_position', and 'Sequence' back as columns
    reads_data_grouped = reads_data_grouped.reset_index()

    # Rename columns accordingly
    reads_data_grouped.columns = [
        '{}_{}'.format(col[0], col[1]) if col[1] != '' else col[0]
        for col in reads_data_grouped.columns
    ]

    # Extract 5-mer from the 7-mer sequences
    reads_data_grouped['Front_5mer'] = reads_data_grouped['Sequence'].str[
        0:5
    ]  # Position 1 to 5 (0-based indexing)
    reads_data_grouped['Middle_5mer'] = reads_data_grouped['Sequence'].str[
        1:6
    ]  # Position 2 to 6 (0-based indexing)
    reads_data_grouped['Back_5mer'] = reads_data_grouped['Sequence'].str[
        2:7
    ]  # Position 3 to 7 (0-based indexing)

    # Create new columns for each DRACH motif
    for motif in dra_ch_motifs:
        reads_data_grouped[motif] = reads_data_grouped['Middle_5mer'].apply(
            lambda x: 1 if x == motif else 0
        )

    for motif in pos_1_5:
        reads_data_grouped[motif] = reads_data_grouped['Front_5mer'].apply(
            lambda x: 1 if x == motif else 0
        )

    for motif in pos_3_7:
        reads_data_grouped[motif] = reads_data_grouped['Back_5mer'].apply(
            lambda x: 1 if x == motif else 0
        )

    # Drop the 'Middle_5mer' column if not needed
    reads_data_grouped = reads_data_grouped.drop(
        columns=['Middle_5mer', 'Front_5mer', 'Back_5mer']
    )

    return reads_data_grouped


def standardise_split_data(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    Standardise training and testing datasets.

    This function scales the feature columns of the training and testing 
    datasets using z-score normalisation.

    Args:
        train_data (pd.DataFrame): Training dataset with features.
        test_data (pd.DataFrame): Testing dataset with features.

    Returns:
        tuple: A tuple containing the standardised training data, 
        testing data, and the fitted scaler.
    """

    # Retrieve column names of data, and exclude columns that we do not want to standardise
    columns_to_standardise = train_data.columns.difference(
        ['gene_id', 'transcript_id', 'transcript_position', 'Sequence', 'label']
    )

    # Initialise the StandardScaler
    scaler = StandardScaler()

    # Scale(standardise)!
    train_data[columns_to_standardise] = scaler.fit_transform(
        train_data[columns_to_standardise]
    )
    test_data[columns_to_standardise] = scaler.transform(
        test_data[columns_to_standardise]
    )

    return train_data, test_data, scaler

def standardise_data(data: pd.DataFrame, standard_scaler_path: str):
    """
    Standardise a given dataset using a pre-trained scaler.

    This function scales the feature columns of the provided dataset using 
    a previously fitted StandardScaler.

    Args:
        data (pd.DataFrame): Dataset with features to be standardised.
        standard_scaler_path (str): Path to the pre-trained StandardScaler model.

    Returns:
        pd.DataFrame: The standardised dataset.
    """

    # Retrieve column names of data, and exclude columns that we do not want to standardise
    columns_to_standardise = data.columns.difference(
        ['gene_id', 'transcript_id', 'transcript_position', 'Sequence', 'label']
    )

    # Initialise the StandardScaler
    scaler = joblib.load(standard_scaler_path)

    # Scale(standardise)!
    data[columns_to_standardise] = scaler.transform(
        data[columns_to_standardise]
    )

    return data
