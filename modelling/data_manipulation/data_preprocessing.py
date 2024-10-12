# Import libraries
import gzip
import json
import pandas as pd

# File paths of datsets 
label_file_path = 'data/data.info.labelled'
data_file_path = 'data/dataset0.json.gz'

# Read in label data
labels = pd.read_csv(label_file_path, sep=',')

# Read and load data from gzipped JSON file
data_bulk = []
with gzip.open(data_file_path, 'rt') as json_lines:
    for json_line in json_lines:
        data_bulk.append(json.loads(json_line))

# Create a label map for easy lookup
label_map = {}
for label in labels.iloc:
    label_map[(str(label['transcript_id']), str(label['transcript_position']))] = label

# Create a datatframe that contains all columns
data_unaggregated = []
for data in data_bulk:
    trns = list(data.keys())[0]
    pos = list(data[trns].keys())[0]
    nucleotides = list(data[trns][pos].keys())[0]
    values = data[trns][pos][nucleotides]

    # Lookup correct label from label_map
    correct_label = label_map[(str(trns), str(pos))]
    gene = correct_label['gene_id']
    score = correct_label['label']

    # Append the unaggregated data to list
    data_unaggregated.append({
        'trns': trns,
        'pos': pos,
        'nucleotides': nucleotides,
        'values': values,
        'gene': gene,
        'score': score
    })

# Convert the unaggregated data into a DataFrame
df_unaggregated = pd.DataFrame(data_unaggregated)

# Save DataFrame to a CSV
df_unaggregated.to_csv('data/unaggregated_data.csv', index=False)