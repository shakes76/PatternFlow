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
    def __init__(self, n_latents, d_latents):
        super(CrossAttention, self).__init__()
        self.q_ln = nn.LayerNorm(1)
        self.kv_ln = nn.LayerNorm(1)
        self.q_l = nn.Linear(n_latents, d_latents)
        self.k_l = nn.Linear(1, d_latents)
        self.v_l = nn.Linear(1, d_latents)
        self.attn = nn.MultiheadAttention(d_latents, 1, batch_first=True)           
        self.o_l = Conv1D(d_latents)

    def forward(self, latent, data):
        q = self.q_l(self.q_ln(latent))
        k = self.k_l(self.kv_ln(data))
        v = self.v_l(self.kv_ln(data))
        attn = self.attn(q, k, v)[0]
        output = self.o_l(attn)
        return output

class SelfAttention(nn.Module):
    """
    The inputs are a latent array, which is the query, key and value. 

    First process inputs through a batch normalisation, before putting them through a linear layer.

    Then put them through the attention block, before one last linear layer
    """
    def __init__(self, n_latents, d_latents):
        super(SelfAttention, self).__init__()
        self.q_ln = nn.LayerNorm(d_latents)
        self.k_ln = nn.LayerNorm(d_latents)
        self.v_ln = nn.LayerNorm(d_latents)
        self.q_l = Conv1D(d_latents)
        self.k_l = Conv1D(d_latents)
        self.v_l = Conv1D(d_latents)
        self.attn = nn.MultiheadAttention(d_latents, 1, batch_first=True)           
        self.o_l = Conv1D(d_latents)

    def forward(self, latent):
        q = self.q_l(self.q_ln(latent))
        k = self.k_l(self.k_ln(latent))
        v = self.v_l(self.v_ln(latent))
        attn = self.attn(q, k, v)[0]
        output = self.o_l(attn)
        return output


class LatentTransformer(nn.Module):
    """
    Consists of a self attention block between two linear layers. 
    """
    pass

class MLP(nn.Module):
    """
    Consists of a layer normalisation, followed by a linear layer with GeLU activation before another linear layer. 
    """
    def __init__(self, d_latents) -> None:
        super(MLP, self).__init__()
        self.ln = nn.LayerNorm(d_latents)
        self.l1 = Conv1D(d_latents)
        self.l2 = Conv1D(d_latents)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.ln(x)
        x = self.gelu(self.l1(x))
        x = self.l2(x)
        return x
        

class Output(nn.Module):
    pass

