import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):

    def __init__(self, max_len=3):
        super().__init__()
        pe = torch.zeros(max_len)      
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        pe = torch.sin(position)

        self.register_buffer('pe', pe)

    def forward(self, word_embeddings):
        return word_embeddings + self.pe
