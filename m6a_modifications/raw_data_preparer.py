import argparse
import gzip
import json

import pandas as pd


def prepare_data(data_file_path: str):
    print(
        f'[raw_data_preparer] - INFO: Initialising the processing of reads: {data_file_path}'
    )

    reads_data = []
    transcript_count = 0
    if data_file_path.endswith('.json.gz'):
        transcript_lines = gzip.open(data_file_path, 'rt')
    elif data_file_path.endswith('json'):
        transcript_lines = open(data_file_path, 'r')
    else:
        raise ValueError('Data file path does not seem to path to a .json.gz file or a .json file.')

    print('[raw_data_preparer] - INFO: Formatting json file, please be patient...')
    for transcript_line in transcript_lines:
        transcript_count += 1
        if transcript_count % 30000 == 0:
            print(
                f'[raw_data_preparer] - DEBUG: Currently processing {transcript_count}th transcript'
            )

        # Retrieve Transcript ID
        transcript_data = json.loads(transcript_line)
        transcript_id = list(transcript_data.keys())[0]

        # Retrieve Transcript Position
        transcript_data = transcript_data[transcript_id]
        position = list(transcript_data.keys())[0]

        # Retrieve Nucleotides
        transcript_data = transcript_data[position]
        nucleotides = list(transcript_data.keys())[0]

        # Retrieve Feature Values
        features = transcript_data[nucleotides]

        for raw_read_values in features:
            reads_data.append(
                {
                    'transcript_id': transcript_id,
                    'transcript_position': position,
                    'Sequence': nucleotides,
                    'Dwelling_Time_1': raw_read_values[0],
                    'SD_1': raw_read_values[1],
                    'Mean_1': raw_read_values[2],
                    'Dwelling_Time_2': raw_read_values[3],
                    'SD_2': raw_read_values[4],
                    'Mean_2': raw_read_values[5],
                    'Dwelling_Time_3': raw_read_values[6],
                    'SD_3': raw_read_values[7],
                    'Mean_3': raw_read_values[8],
                }
            )

    # Save the output as a CSV file
    print(
        '[raw_data_preparer] - INFO: All transcripts processed, converting to CSV (this may take a while)...'
    )
    reads_dataframe = pd.DataFrame(reads_data)
    reads_dataframe.to_csv(f'{data_file_path}.csv', index=False)
    print(f'[raw_data_preparer] - INFO: Reads have been saved: {data_file_path}.csv')

    return reads_dataframe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process transcript data from the raw compressed dataset json file.'
    )
    parser.add_argument(
        '--data_file_path', type=str, help='Path to the gzipped dataset json file.'
    )

    args = parser.parse_args()
    prepare_data(args.data_file_path)
