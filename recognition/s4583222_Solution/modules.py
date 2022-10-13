# COMP3710 Pattern Recognition Lab Assignment
# By Thomas Jellett (s4583222)
# HARD DIFFICULTY
# Create a generative model of the OASIS brain using stable diffusion that
# has a â€œreasonably clear image.â€

# File: modules.py
# Description: My model

#Images are 3x256x256
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 64


#--------------UNET------------------#

class PositionEmbedding(nn.Module):
    def __init__(self, time_emb):
        super().__init__()
        self.time_emb = time_emb
        self.relu = nn.ReLU()
        self.linear = nn.Linear(time_emb, time_emb)
    
    def forward(self, time):
        # time = time.to(DEVICE)
        device = time.device
        half = self.time_emb // 2
        embed = math.log(10000) / (half - 1)
        embed = torch.exp(torch.arange(half, device=device) * - embed)
        embed = time[:, None] * embed[None, :]
        position_encode = torch.cat((embed.sin(), embed.cos()), dim = -1)
        position_encode = self.linear(position_encode)
        position_encode = self.relu(position_encode)
        return position_encode

class Block(nn.Module):
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
    def __init__(self, in_chan = 3, out_chan = 3, device = DEVICE):
        super().__init__()
        self.device = device
        time_dimension = 32
        #Network 16 32 64 128 256

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.time_embedding = PositionEmbedding(time_dimension)

        self.first_level = Block(in_chan, 16, time_dimension)
        self.second_level = Block(16, 32, time_dimension)
        self.third_level = Block(32, 64, time_dimension)
        self.fourth_level = Block(64, 128, time_dimension)

        self.bottle_neck = Block(128, 256, time_dimension)

        self.fifth_level_up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)       
        self.fifth_level = Block(256, 128, time_dimension)
        self.sixth_level_up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.sixth_level = Block(128, 64, time_dimension)
        self.seventh_level_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.seventh_level = Block(64, 32, time_dimension)
        self.eigth_level_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.eigth_level = Block(32, 16, time_dimension)

        self.last_conv = nn.Conv2d(16, out_chan, kernel_size=1)

    def forward(self, x, t):
        t = self.time_embedding(t)

        x1 = self.first_level(x, t)
        x1_skip = x1
        
        x2 = self.pool(x1)
        x2 = self.second_level(x2, t)
        x2_skip = x2

        x3 = self.pool(x2)
        x3 = self.third_level(x3, t)
        x3_skip = x3

        x4 = self.pool(x3)
        x4 = self.fourth_level(x4, t)
        x4_skip = x4

        x_bottle = self.pool(x4)
        x_bottle = self.bottle_neck(x_bottle, t)

        x5 = self.fifth_level_up(x_bottle)        
        x5 = torch.cat([x4_skip, x5], dim = 1)
        x5 = self.fifth_level(x5, t)
        
        x6 = self.sixth_level_up(x5) 
        x6 = torch.cat([x3_skip, x6], dim = 1)
        x6 = self.sixth_level(x6, t)

        x7 = self.seventh_level_up(x6) 
        x7 = torch.cat([x2_skip, x7], dim = 1)
        x7 = self.seventh_level(x7, t)
        
        x8 = self.eigth_level_up(x7) 
        x8 = torch.cat([x1_skip, x8], dim = 1)
        x8 = self.eigth_level(x8, t) 
        
        output_x = self.last_conv(x8)
        
        return output_x 

# class DownBlock(nn.Module):
#     def __init__(self, in_chan, out_chan, time_emb):
#         super().__init__()    
#         self.conv_1 = nn.Sequential(
#             nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_chan),
#             nn.ReLU(),
#         )
#         self.conv_2 = nn.Sequential(
#             nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_chan),
#             nn.ReLU(),
#         )
#         self.relu = nn.ReLU()
#         self.time = nn.Linear(time_emb, out_chan)
    
#     def forward(self, x, t):
#         out = self.conv_1(x)
#         time_embedded = self.time(t)
#         time_embedded = self.relu(time_embedded)
#         time_embedded = time_embedded[(...,) + (None,) * 2]
#         out = out + time_embedded
#         out = self.conv_2(out)
#         return out

# class UpBlock(nn.Module):
#     def __init__(self, in_chan, out_chan, time_emb):
#         super().__init__()
#         self.conv_1 = nn.Sequential(
#             nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_chan),
#             nn.ReLU(),
#         )
#         self.conv_2 = nn.Sequential(
#             nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_chan),
#             nn.ReLU(),
#         )
#         self.relu = nn.ReLU()
#         self.time = nn.Linear(time_emb, out_chan)
    
#     def forward(self, x, t):
#         out = self.conv_1(x)
#         time_embedded = self.time(t)
#         time_embedded = self.relu(time_embedded)
#         time_embedded = time_embedded[(...,) + (None,) * 2]
#         out = out + time_embedded
#         out = self.conv_2(out)
#         return out

# class UNETModel(nn.Module):
#     def __init__(self, in_chan = 3, out_chan = 3, time_dimension = 32, device = DEVICE):
#         super().__init__()
#         self.device = device
#         self.time_dimension = time_dimension

#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.time_embedding = PositionEmbedding(time_dimension)
        
        
#         self.first_level = DownBlock(in_chan, 16, time_dimension)
#         self.second_level = DownBlock(16, 32, time_dimension)
#         self.third_level = DownBlock(32, 64, time_dimension)
#         self.fourth_level = DownBlock(64, 128, time_dimension)
#         self.bottle_neck = DownBlock(128, 256, time_dimension)
#         self.up_sample = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.fifth_level = UpBlock(256, 128, time_dimension)
#         self.up_sample_5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) 
#         self.sixth_level = UpBlock(128, 64, time_dimension)
#         self.up_sample_6 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
#         self.seventh_level = UpBlock(64, 32, time_dimension)
#         self.up_sample_7 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
#         self.eigth_level = UpBlock(32, 16, time_dimension)
#         self.last_conv = nn.Conv2d(16, out_chan, kernel_size=1)

#     def forward(self, x, t):
#         t = self.time_embedding(t)

#         x1 = self.first_level(x, t)
#         x1_skip = x1
        
#         x2 = self.pool(x1)
#         x2 = self.second_level(x2, t)
#         x2_skip = x2

#         x3 = self.pool(x2)
#         x3 = self.third_level(x3, t)
#         x3_skip = x3

#         x4 = self.pool(x3)
#         x4 = self.fourth_level(x4, t)
#         x4_skip = x4

#         x_bottle = self.pool(x4)
#         x_bottle = self.bottle_neck(x_bottle, t)
#         x5 = self.up_sample(x_bottle)
        
#         x5 = torch.cat([x4_skip, x5], dim=1)
#         x5 = self.fifth_level(x5, t)
#         x6 = self.up_sample_5(x5)

#         x6 = torch.cat([x3_skip, x6], dim = 1)
#         x6 = self.sixth_level(x6, t) 
#         x7 = self.up_sample_6(x6)

#         x7 = torch.cat([x2_skip, x7], dim = 1)
#         x7 = self.seventh_level(x7, t)
#         x8 = self.up_sample_7(x7) 

#         x8 = torch.cat([x1_skip, x8], dim = 1)
#         x8 = self.eigth_level(x8, t) 
        
#         output_x = self.last_conv(x8)
        
#         return output_x 
