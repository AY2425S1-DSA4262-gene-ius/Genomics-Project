import torch
import torch.nn as nn
from m6a_powernet.subnet import SubNet

class M6APowerNet(nn.Module):
    def __init__(self, input_size=3):
        super().__init__()

        self.signal_length_subnet = SubNet(max_len=input_size)
        self.signal_sd_subnet = SubNet(max_len=input_size)
        self.signal_mean_subnet = SubNet(max_len=input_size)

        self.seven_mer_embedder = nn.Embedding(num_embeddings=288, embedding_dim=16)
        self.sever_mer_layer = nn.Linear(16, 2)

        self.global_layer1 = nn.Linear(11, 32)
        self.global_batch_norm1 = nn.BatchNorm1d(32)
        self.global_layer2 = nn.Linear(32, 8)
        self.global_batch_norm2 = nn.BatchNorm1d(8)
        self.global_layer3 = nn.Linear(8, 1)

        self.activation = nn.ReLU()

    
    def forward(self, input):
        
        # input = [torch.FloatTensor([[[1],[2],[3]], [[2],[2],[2]]]), torch.FloatTensor([[[4],[5],[6]], [[2],[2],[2]]]), torch.FloatTensor([[[7],[8],[9]], [[4],[1],[1]]]), torch.tensor([15])]
        signal_length, signal_sd, signal_mean, seven_mer = input

        signal_length_subnet_output = self.signal_length_subnet(signal_length)
        signal_sd_subnet_output = self.signal_sd_subnet(signal_sd)
        signal_mean_subnet_output = self.signal_mean_subnet(signal_mean)
        seven_mer_output = self.sever_mer_layer(self.seven_mer_embedder(seven_mer)).expand(signal_length_subnet_output.shape[0], -1)

        combined = torch.cat((signal_length_subnet_output, signal_sd_subnet_output, signal_mean_subnet_output, seven_mer_output), dim=1)

        output = self.activation(self.global_batch_norm1(self.global_layer1(combined)))
        output = self.activation(self.global_batch_norm2(self.global_layer2(output)))
        output = self.global_layer3(output)

        probabilities = torch.sigmoid(output)
        return 1 - torch.prod(1 - probabilities)
