import torch.nn as nn

from m6a_powernet.attention import SingleHeadAttention
from m6a_powernet.positional_encoder import PositionalEncoder

class SubNet(nn.Module): 
    def __init__(self, d_model=1, max_len=3):
        super().__init__()

        self.positional_encoder = PositionalEncoder(max_len=max_len)
        self.self_attention = SingleHeadAttention(d_model=d_model, bias=False)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, embeddings):

        positionally_encoded_embeddings = self.positional_encoder(embeddings)
        self_attention_output = self.self_attention(positionally_encoded_embeddings)

        # Residual connection
        residual_connected_output = positionally_encoded_embeddings + self_attention_output

        return self.output(residual_connected_output)
