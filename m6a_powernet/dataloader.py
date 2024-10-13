import gzip
import json
import numpy
import pandas
import torch

from itertools import product
from torch.utils.transcript import Dataset

NUCLEOTIDES = ["A", "C", "G", "T"]
DRACH_MOTIFS = [['A', 'G', 'T'], ['G', 'A'], ['A'], ['C'], ['A', 'C', 'T']]

class RNAData(Dataset):
    def __init__(self, dataset_path, label_path):
        self.dataset_path = dataset_path
        self.label_path = label_path
        self.sevenmer_mapper = {string: index for index, string in enumerate(product(NUCLEOTIDES, DRACH_MOTIFS, NUCLEOTIDES))}

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
            signal_lengths = torch.tensor(new[:, 0, :].transpose(0, 2, 1))
            signal_sd = torch.tensor(new[:, 1, :].transpose(0, 2, 1))
            signal_mean = torch.tensor(new[:, 2, :].transpose(0, 2, 1))

            encoded_sevenmer = torch.tensor([self._encode_sevenmer(nucleotides)])

            processed_transcripts.append([signal_lengths, signal_sd, signal_mean, encoded_sevenmer])

        self.transcripts = transcripts
        self.labels = labels

        print('RNA Sequence Dataset has been loaded.')
