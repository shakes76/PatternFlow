import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256


class PositionEmbedding(nn.Module):
    """
    Takes a tensor of size (batch size, 1) and transforms it into a tensor of size (batch size, time_emb)
    Acknowledgment From:   
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=592aa765
    """
    def __init__(self, time_emb):
        super().__init__()
        self.time_emb = time_emb
        self.relu = nn.ReLU()
        self.linear = nn.Linear(time_emb, time_emb)
    
    def forward(self, time):
        # time = time.to(DEVICE)
        device = time.device
        embed = math.log(10000) / ((self.time_emb // 2) - 1)
        embed = torch.exp(torch.arange((self.time_emb // 2), device=device) * - embed)
        embed = time[:, None] * embed[None, :]
        position_encode = torch.cat((embed.sin(), embed.cos()), dim = -1)
        position_encode = self.linear(position_encode)
        position_encode = self.relu(position_encode)
        return position_encode

class Block(nn.Module):
    """
    This is a residual block of the UNET. Both Contraction and Expansion path uses this block.
    """
    def __init__(self, in_chan, out_chan, time_emb):
        super().__init__()    
        self.relu = nn.ReLU()
        self.time = nn.Linear(time_emb, out_chan)
        self.conv_1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)  
        self.conv_2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_chan)
        self.bnorm2 = nn.BatchNorm2d(out_chan)               
    
    def forward(self, x, t):
        out = self.conv_1(x)
        out = self.bnorm1(out)
        out = self.relu(out)
        time_embedding = self.time(t)
        time_embedding = self.relu(time_embedding)
        time_embedding = time_embedding[(...,) + (None,) * 2]
        out = out + time_embedding
        out = self.conv_2(out)
        out = self.bnorm2(out)
        out = self.relu(out)
        return out


class UNETModel(nn.Module):
    """
    Modified UNET model with Sinusoidal Position Embedding
    """
    def __init__(self, in_chan = 3, out_chan = 3, device = DEVICE):
        super().__init__()
        self.device = device
        time_dimension = 32
        #Network 64 128 256 512 1024

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.time_embedding = PositionEmbedding(time_dimension)

        #Contraction Path
        self.first_level = Block(in_chan, 64, time_dimension)
        self.second_level = Block(64, 128, time_dimension)
        self.third_level = Block(128, 256, time_dimension)
        self.fourth_level = Block(256, 512, time_dimension)

        #Bottle neck (floor)
        self.bottle_neck = Block(512, 1024, time_dimension)

        #Expansion path
        self.fifth_level_up = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)       
        self.fifth_level = Block(1024, 512, time_dimension)
        self.sixth_level_up = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.sixth_level = Block(512, 256, time_dimension)
        self.seventh_level_up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.seventh_level = Block(256, 128, time_dimension)
        self.eigth_level_up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.eigth_level = Block(128, 64, time_dimension)

        #Final Conv
        self.last_conv = nn.Conv2d(64, out_chan, kernel_size=1)

    def forward(self, x, t):
        t = self.time_embedding(t)

        x1 = self.first_level(x, t)#First residual block
        x1_skip = x1#Skip connection
        
        x2 = self.pool(x1)#Maxpool
        x2 = self.second_level(x2, t)#Second residual block
        x2_skip = x2

        x3 = self.pool(x2)
        x3 = self.third_level(x3, t)#Third residual block
        x3_skip = x3

        x4 = self.pool(x3)
        x4 = self.fourth_level(x4, t)#Fourth residual block
        x4_skip = x4

        x_bottle = self.pool(x4)
        x_bottle = self.bottle_neck(x_bottle, t)#Double conv with time embedding

        x5 = self.fifth_level_up(x_bottle)#Upsample        
        x5 = torch.cat([x4_skip, x5], dim = 1)#Concat with skip connection
        x5 = self.fifth_level(x5, t)#Fifth residual block
        
        x6 = self.sixth_level_up(x5) 
        x6 = torch.cat([x3_skip, x6], dim = 1)
        x6 = self.sixth_level(x6, t)#Sixth residual block

        x7 = self.seventh_level_up(x6) 
        x7 = torch.cat([x2_skip, x7], dim = 1)
        x7 = self.seventh_level(x7, t)#Seventh residual block
        
        x8 = self.eigth_level_up(x7) 
        x8 = torch.cat([x1_skip, x8], dim = 1)
        x8 = self.eigth_level(x8, t)#Eigth residual block 
        
        output_x = self.last_conv(x8)#Final convolution, output image is the same dimensions as input
        
        return output_x 