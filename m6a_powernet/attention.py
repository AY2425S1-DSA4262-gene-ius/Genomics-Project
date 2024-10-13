import torch
from torch import nn

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int, bias: bool = False):
        super().__init__()

        self.query = nn.Linear(d_model, d_model, bias=bias)
        self.key = nn.Linear(d_model, d_model, bias=bias)
        self.value = nn.Linear(d_model, d_model, bias=True)

        self.softmax = nn.Softmax(dim=-1)

        self.output = nn.Linear(d_model, d_model)
        self.attn = None

    def forward(self, encoding: torch.Tensor):
        query = self.query(encoding)
        key = self.key(encoding)
        value = self.value(encoding)

        scores = torch.matmul(query, key.transpose(-1, -2))
        attn = self.softmax(scores)

        x = torch.matmul(attn, value)

        self.attn = attn.detach()

        return self.output(x)
