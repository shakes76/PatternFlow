from inspect import Parameter
from turtle import forward
import torch
import torch.nn as nn

def Conv1D(out_dim):
    return nn.Linear(out_dim, out_dim)
 
class CrossAttention(nn.Module):
    """
    The inputs are a latent array and the actual data. The latent array is the query and the data forms the key and value.

    First process inputs through a batch normalisation, before putting them through a linear layer.

    Then put them through the attention block, before one last linear layer
    """
    def __init__(self, d_latents):
        super(CrossAttention, self).__init__()
        self.q_ln = nn.LayerNorm(d_latents)
        self.kv_ln = nn.LayerNorm(1)
        self.q_l = nn.Linear(d_latents, d_latents)
        self.k_l = nn.Linear(1, d_latents)
        self.v_l = nn.Linear(1, d_latents)
        self.attn = nn.MultiheadAttention(d_latents, 1, batch_first=True)           
        self.o_l = nn.Linear(d_latents, d_latents)

    def forward(self, latent, data):
        q = self.q_ln(latent)
        q = self.q_l(q)
        k = self.kv_ln(data)
        k = self.k_l(k)
        v = self.kv_ln(data)
        v = self.v_l(v)
        attn = self.attn(q, k, v)[0]
        output = self.o_l(attn)
        return output

class SelfAttention(nn.Module):
    """
    The inputs are a latent array, which is the query, key and value. 

    First process inputs through a batch normalisation, before putting them through a linear layer.

    Then put them through the attention block, before one last linear layer
    """
    def __init__(self, d_latents):
        super(SelfAttention, self).__init__()
        self.q_ln = nn.LayerNorm(d_latents)
        self.k_ln = nn.LayerNorm(d_latents)
        self.v_ln = nn.LayerNorm(d_latents)
        self.q_l = nn.Linear(d_latents, d_latents)
        self.k_l = nn.Linear(d_latents, d_latents)
        self.v_l = nn.Linear(d_latents, d_latents)
        self.attn = nn.MultiheadAttention(d_latents, 1, batch_first=True)           
        self.o_l = nn.Linear(d_latents, d_latents)

    def forward(self, latent):
        q = self.q_ln(latent)
        q = self.q_l(q)
        k = self.k_ln(latent)
        k = self.k_l(k)
        v = self.v_ln(latent)
        v = self.v_l(v)
        attn = self.attn(q, k, v)[0]
        output = self.o_l(attn)
        return output


class LatentTransformer(nn.Module):
    """
    Consists of a self attention block between two linear layers. 
    """
    def __init__(self, d_latents, depth) -> None:
        super(LatentTransformer, self).__init__()
        self.ff = nn.ModuleList([MLP(d_latents) for _ in range(depth)])
        self.sa = nn.ModuleList([SelfAttention(d_latents) for _ in range(depth)])
        self.depth = depth

    def forward(self, x):
        latent = x
        for i in range(self.depth):
            latent = self.sa[i](latent) + latent
            latent = self.ff[i](latent)
        return latent

class MLP(nn.Module):
    """
    Consists of a layer normalisation, followed by a linear layer with GeLU activation before another linear layer. 
    """
    def __init__(self, d_latents) -> None:
        super(MLP, self).__init__()
        self.ln = nn.LayerNorm(d_latents)
        self.l1 = nn.Linear(d_latents, d_latents)
        self.l2 = nn.Linear(d_latents, d_latents)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.ln(x)
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x

class Block(nn.Module):
    """
    A Cross attend block followed by a Latent Transformer block.
    """
    def __init__(self, d_latents) -> None:
        super(Block, self).__init__()
        self.ca = CrossAttention(d_latents)
        self.ff = MLP(d_latents)

    def forward(self, x, data):
        attn = self.ca(x, data)
        x = attn + x
        x = self.ff(x)
        return x
        

class Output(nn.Module):
    def __init__(self, d_latents, n_classes=2) -> None:
        super(Output, self).__init__()
        self.project = nn.Linear(d_latents, n_classes)
    
    def forward(self, x):
        average = torch.mean(x, dim=1)
        logits = self.project(average)
        return logits


class Perciever(nn.Module):
    def __init__(self, n_latents, d_latents, transformer_depth, n_cross_attends) -> None:
        super(Perciever, self).__init__()
        self.depth = n_cross_attends

        # Initialise the latent array
        latent = torch.empty(n_latents, d_latents)
        nn.init.trunc_normal_(latent,std=0.02)
        self.latent=nn.Parameter(latent)

        self.ca = nn.ModuleList([Block(d_latents) for _ in range(n_cross_attends)])
        self.lt = nn.ModuleList([LatentTransformer(d_latents, transformer_depth) for _ in range(n_cross_attends)])
        self.op = Output(d_latents)


    def forward(self, data):
        b, _, _, _ = data.size()
        x = torch.stack([self.latent for _ in range(b)])
        for i in range(self.depth):
            x = self.ca[i](x, torch.flatten(data, start_dim=1)[:, :, None])
            x = self.lt[i](x)
        output = self.op(x)
        return output
