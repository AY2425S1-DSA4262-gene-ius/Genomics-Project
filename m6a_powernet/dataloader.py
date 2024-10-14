import gzip
import json
import numpy
import pandas
import torch

from itertools import product
from torch.utils.data import Dataset

NUCLEOTIDES = ["A", "C", "G", "T"]
DRACH_MOTIFS = [['A', 'G', 'T'],
                ['G', 'A'],
                ['A'],
                ['C'],
                ['A', 'C', 'T']]

class RNAData(Dataset):
    def __init__(self, dataset_path, label_path):
        self.dataset_path = dataset_path
        self.label_path = label_path
        self.sevenmer_mapper = {''.join(sevenmer): index for index, sevenmer in enumerate(product(NUCLEOTIDES, *DRACH_MOTIFS, NUCLEOTIDES))}
        self.processed_transcripts, self.processed_labels = self.load_data()

    def _read_data(self):
        transcripts = []
        with gzip.open(self.dataset_path, 'rt') as dataset:
            for transcript in dataset:
                transcripts.append(json.loads(transcript))

        labels = pandas.read_csv(self.label_path, sep=',')

        return transcripts, labels

    def _encode_sevenmer(self, sevenmer):
        return self.sevenmer_mapper[sevenmer]

    def load_data(self):
        print('Loading the RNA Sequence Dataset...')

        processed_transcripts = []
        processed_labels = []

        transcripts, labels = self._read_data()

        for transcript in transcripts:
            transcript_name = list(transcript.keys())[0]
            position = list(transcript[transcript_name].keys())[0]
            nucleotides = list(transcript[transcript_name][position].keys())[0] # Sevenmer
            values = transcript[transcript_name][position][nucleotides]

            data_np = numpy.array(values)
            # Breaking up the reads from 1 by 9 to 3 by 3
            # Then, transpose the 3 by 3 in such a way that each row is the same data type from different 5mer, rather than different data types from the same 5mer.
            new = data_np.reshape(-1, 3, 3).transpose(0, 2, 1)

            # From the N by 3 by 3, we split it into 3 tensors, such that each tensor stores N by 1 by 3, where each tensor is a data type.
            # Then, transpose the 1 by 3 to become 3 by 1, so that the model can use.
            signal_lengths = torch.FloatTensor(new[:, 0:1, :].transpose(0, 2, 1))
            signal_sd = torch.FloatTensor(new[:, 1:2, :].transpose(0, 2, 1))
            signal_mean = torch.FloatTensor(new[:, 2:3, :].transpose(0, 2, 1))

            encoded_sevenmer = torch.tensor([self._encode_sevenmer(nucleotides)])

            processed_transcripts.append([signal_lengths, signal_sd, signal_mean, encoded_sevenmer])

        for label in labels.iloc:
            processed_labels.append(torch.FloatTensor(label['label']))

        # Train test split by gene_id
        unique_gene_ids = labels['gene_id'].unique()
        numpy.random.shuffle(unique_gene_ids)
        split_index = int(len(unique_gene_ids) * 0.8)
        train_gene_ids = unique_gene_ids[:split_index]
        test_gene_ids = unique_gene_ids[split_index:]

        

        print('RNA Sequence Dataset has been loaded.')
        return processed_transcripts, processed_labels

    def __len__(self):
        return len(self.processed_transcripts)

    def __getitem__(self, idx):
        transcript_data = self.processed_transcripts[idx]
        label = self.processed_labels[idx]
        
        return transcript_data, label
