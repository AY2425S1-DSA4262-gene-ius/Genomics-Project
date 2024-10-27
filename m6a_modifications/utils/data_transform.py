import pandas as pd
from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None

# Possible middle 5-mers
dra_ch_motifs = [
    'AAACA',
    'AAACC',
    'AAACT',
    'AGACA',
    'AGACC',
    'AGACT',
    'GAACA',
    'GAACC',
    'GAACT',
    'GGACA',
    'GGACC',
    'GGACT',
    'TAACA',
    'TAACC',
    'TAACT',
    'TGACA',
    'TGACC',
    'TGACT',
]

# Possible front 5-mers
pos_1_5 = [
    'GAAAC',
    'AAAAC',
    'CAAAC',
    'TAAAC',
    'GAGAC',
    'AAGAC',
    'CAGAC',
    'TAGAC',
    'GGAAC',
    'AGAAC',
    'CGAAC',
    'TGAAC',
    'GGGAC',
    'AGGAC',
    'CGGAC',
    'TGGAC',
    'GTAAC',
    'ATAAC',
    'CTAAC',
    'TTAAC',
    'GTGAC',
    'ATGAC',
    'CTGAC',
    'TTGAC',
]

# Possible back 5-mers
pos_3_7 = [
    'AACAG',
    'AACAA',
    'AACAC',
    'AACAT',
    'AACCG',
    'AACCA',
    'AACCC',
    'AACCT',
    'AACTG',
    'AACTA',
    'AACTC',
    'AACTT',
    'GACAG',
    'GACAA',
    'GACAC',
    'GACAT',
    'GACCG',
    'GACCA',
    'GACCC',
    'GACCT',
    'GACTG',
    'GACTA',
    'GACTC',
    'GACTT',
]


def engineer_features(reads_data: pd.DataFrame):
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
    )

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


def standardise_data(train_data: pd.DataFrame, test_data: pd.DataFrame):
    # Retrieve column names of data, and exclude columns that we do not want to standardise
    columns_to_standardise = train_data.columns.difference(
        ['gene_id', 'transcript_id', 'transcript_position', 'Sequence', 'label']
    )

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Scale(standardise)!
    train_data[columns_to_standardise] = scaler.fit_transform(
        train_data[columns_to_standardise]
    )
    test_data[columns_to_standardise] = scaler.transform(
        test_data[columns_to_standardise]
    )

    return train_data, test_data, scaler
