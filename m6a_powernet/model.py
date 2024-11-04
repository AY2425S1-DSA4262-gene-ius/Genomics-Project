import torch
import torch.nn as nn
from m6a_powernet.subnet import SubNet

class M6APowerNet(nn.Module):
    def __init__(self, input_size=3):
        super().__init__()

        self.signal_length_subnet = SubNet(max_len=input_size)
        self.signal_sd_subnet = SubNet(max_len=input_size)
        self.signal_mean_subnet = SubNet(max_len=input_size)

        self.seven_mer_embedder = nn.Embedding(num_embeddings=288, embedding_dim=64)
        self.sever_mer_layer1 = nn.Linear(64, 32)
        self.sever_mer_layer_norm1 = nn.LayerNorm(32)
        self.sever_mer_layer2 = nn.Linear(32, 3)

        self.global_layer1 = nn.Linear(12, 32)
        self.global_layer_norm1 = nn.LayerNorm(32)
        self.global_layer2 = nn.Linear(32, 64)
        self.global_layer_norm2 = nn.LayerNorm(64)
        self.global_layer3 = nn.Linear(64, 32)
        self.global_layer_norm3 = nn.LayerNorm(32)
        self.global_layer4 = nn.Linear(32, 8)
        self.global_layer_norm4 = nn.LayerNorm(8)
        self.global_layer5 = nn.Linear(8, 1)

        self.dropout = nn.Dropout(0.3)

        self.activation = nn.GELU()

    
    def forward(self, input):
        
        # input = [torch.FloatTensor([[[1],[2],[3]], [[2],[2],[2]]]), torch.FloatTensor([[[4],[5],[6]], [[2],[2],[2]]]), torch.FloatTensor([[[7],[8],[9]], [[4],[1],[1]]]), torch.tensor([15])]
        signal_length, signal_sd, signal_mean, seven_mer = input

        # SubNet outputs
        signal_length_subnet_output = self.signal_length_subnet(signal_length)
        signal_sd_subnet_output = self.signal_sd_subnet(signal_sd)
        signal_mean_subnet_output = self.signal_mean_subnet(signal_mean)
        seven_mer_output = self.activation(self.sever_mer_layer_norm1(self.sever_mer_layer1(self.seven_mer_embedder(seven_mer)).expand(signal_length_subnet_output.shape[0], -1)))
        seven_mer_output = self.dropout(seven_mer_output)
        seven_mer_output = self.sever_mer_layer2(seven_mer_output)


        combined = torch.cat((signal_length_subnet_output, signal_sd_subnet_output, signal_mean_subnet_output, seven_mer_output), dim=1)

        global_output = self.activation(self.global_layer_norm1(self.global_layer1(combined)))
        global_output = self.dropout(global_output)
        global_output = self.activation(self.global_layer_norm2(self.global_layer2(global_output)))
        global_output = self.dropout(global_output)
        global_output = self.activation(self.global_layer_norm3(self.global_layer3(global_output)))
        global_output = self.dropout(global_output)
        global_output = self.activation(self.global_layer_norm4(self.global_layer4(global_output)))
        global_output = self.dropout(global_output)
        output = self.global_layer5(global_output)

        probabilities = torch.sigmoid(output)
        return 1 - torch.prod(1 - probabilities)
