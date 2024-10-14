import gzip
import json
import numpy
import pandas
import torch

from itertools import product
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

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
        self.processed_transcripts, self.processed_labels, self.train_indices, self.test_indices = self._load_and_split_data()
        self.train_mode = True # Default is train mode

    def train_mode(self):
        self.train_mode = True
    
    def test_mode(self):
        self.train_mode = False

    def _read_data(self):
        transcripts = []
        with gzip.open(self.dataset_path, 'rt') as dataset:
            for transcript in dataset:
                transcripts.append(json.loads(transcript))

        labels = pandas.read_csv(self.label_path, sep=',')

        return transcripts, labels

    def _encode_sevenmer(self, sevenmer):
        return self.sevenmer_mapper[sevenmer]

    def _load_and_split_data(self):
        print('Loading the RNA Sequence Dataset...')

        processed_transcripts = []
        processed_labels = []

        transcripts, labels = self._read_data()

        # Train test split by gene_id
        unique_gene_ids = labels['gene_id'].unique()
        numpy.random.shuffle(unique_gene_ids)
        split_index = int(len(unique_gene_ids) * 0.8)
        train_gene_ids = set(unique_gene_ids[:split_index])
        test_gene_ids = set(unique_gene_ids[split_index:])

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

        train_indices = []
        test_indices = []

        for index, label in enumerate(labels.iloc):
            processed_labels.append(torch.tensor(label['label']))
            if label['gene_id'] in train_gene_ids:
                train_indices.append(index)
            elif label['gene_id'] in test_gene_ids:
                test_indices.append(index)
            else:
                raise ValueError('How did you get this error? It is literally not possible.')

        print('RNA Sequence Dataset has been loaded.')
        return processed_transcripts, processed_labels, train_indices, test_indices

    def data_loader(self):
        labels_tracker = {}
        for _, label in self:
            int_label = int(label.item())
            labels_tracker[int_label] = labels_tracker.get(int_label, 0) + 1

        total_datapoints = sum(labels_tracker.values())
        label_weights = {
            label: (total_datapoints - instance_count) / total_datapoints
            for label, instance_count in labels_tracker.items()
        }

        weights = list(map(lambda _, label: label_weights[int(label.item())], self))
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        loader = DataLoader(self, sampler=sampler, batch_size=1)

        return loader

    def __len__(self):
        return len(self.train_indices) if self.train_mode else len(self.test_indices)

    def __getitem__(self, idx):
        data_index = self.train_indices[idx] if self.train_mode else self.test_indices[idx]
        transcript_data = self.processed_transcripts[data_index]
        label = self.processed_labels[data_index]
        
        return transcript_data, label
