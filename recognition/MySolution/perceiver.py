import torch
from torch import nn, einsum
from einops import rearrange, repeat

class Attention(nn.Module):

    # query dimension for cross attention purposes
    def __init__(self, q_dim, context_dim=None, heads=8, dim_head=8, dropout=0.0):
        super().__init__()

        # self attention
        if context_dim is None:
            context_dim = q_dim

        # scaling for attention based on the formula sqrt(dimension)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.query = nn.Linear(q_dim, dim_head * heads, bias=False)
        # taking account for crossattention, input_dim == byte array dim
        self.key = nn.Linear(context_dim, dim_head * heads, bias=False)
        self.val = nn.Linear(context_dim, dim_head * heads, bias=False)

        self.out = nn.Sequential(
            # for cross attention results to latent_dim/query_dim
            nn.Linear(dim_head * heads, q_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        # self attention
        if context is None:
            context = x

        # forward pass q(x), (k, v)(context)
        # for cross attention purposes, context is input data, x is latent
        q, k, v = self.query(x), self.key(context), self.val(context)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))

        attention = einsum("b q d, b k d -> b q k", q, k) * self.scale
        # the same dimension is value and key, therefore perform matrix 
        # multiplication on k, v dimensions
        # softmax(qk/sqrt(dimension)) * v
        attention = einsum("b q k, b k d -> b q d",
                           torch.softmax(attention, dim=-1), v)
        
        # rearrange to match the input
        out = rearrange(attention, "(b h) n d -> b n (h d)", h=self.heads)
        return self.out(out)