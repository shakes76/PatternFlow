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

#-------The Forward Process----------#
# Noising the image

class Diffusion:
    def __init__(self, timesteps = 1000, start_beta = 0.0001, end_beta = 0.02, img_size = 256, device = DEVICE):
        self.timesteps = timesteps
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.img_size = img_size
        self.device = device

        self.beta = self.linear_beta().to(device)
        self.alpha = 1. - self.beta #alpha is 1 - beta 
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) #alpha hat is the cumulative product of alpha

    def linear_beta(self):
        return torch.linspace(self.start_beta, self.end_beta, self.timesteps)

    def forward_process(self, x, t):
        sqrt_alpha_hat_term = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat_term = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        x_t = sqrt_alpha_hat_term * x + sqrt_one_minus_alpha_hat_term * epsilon
        return x_t, epsilon

    def sample_the_timestep(self, n):
        return torch.randint(low = 1, high=self.timesteps, size = (n,))
    
    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x_t = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.timesteps)), position = 0):
                t = (torch.ones(n)*i).long().to(self.device)
                pred_noise = model(x_t, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x_t)
                else:
                    noise = torch.zeros_like(x_t)
                x_t_minus_one = 1/torch.sqrt(alpha) * (x_t - ((1 - alpha)/(torch.sqrt(1 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise
        model.train()
        x__t_minus_one = (x_t_minus_one.clamp(-1, 1) + 1)/2
        x__t_minus_one = (x__t_minus_one * 255).type(torch.uint8)
        return x__t_minus_one




#--------------UNET------------------#

class PositionEmbedding(nn.Module):
    def __init__(self, time_emb):
        super().__init__()
        self.time_emb = time_emb
        self.relu = nn.ReLU()
        self.linear = nn.Linear(time_emb, time_emb)
    
    def forward(self, time):
        time = time.to(DEVICE)
        embed = math.log(10000) / ((self.time_emb//2) - 1)
        embed = torch.exp(torch.arange(self.time_emb//2, device=DEVICE) * - embed)
        embed = time[:, None] * embed[None, :]
        position_encode = torch.cat((embed.sin(), embed.cos()), dim = 1)
        position_encode = self.linear(position_encode)
        position_encode = self.relu(position_encode)
        return position_encode

class DownBlock(nn.Module):
    def __init__(self, in_chan, out_chan, time_emb):
        super().__init__()    
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()
        self.time = nn.Linear(time_emb, out_chan)
    
    def forward(self, x, t):
        out = self.conv_1(x)
        time_embedded = self.time(t)
        time_embedded = self.relu(time_embedded)
        time_embedded = time_embedded[(...,) + (None,) * 2]
        out = out + time_embedded
        out = self.conv_2(out)
        return out

class UpBlock(nn.Module):
    def __init__(self, in_chan, out_chan, time_emb):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()
        self.time = nn.Linear(time_emb, out_chan)
    
    def forward(self, x, t):
        out = self.conv_1(x)
        time_embedded = self.time(t)
        time_embedded = self.relu(time_embedded)
        time_embedded = time_embedded[(...,) + (None,) * 2]
        out = out + time_embedded
        out = self.conv_2(out)
        return out


class UNETModel(nn.Module):
    def __init__(self, in_chan = 3, out_chan = 3, time_dimension = 32, device = DEVICE):
        super().__init__()
        self.device = device
        self.time_dimension = time_dimension

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.time_embedding = PositionEmbedding(time_dimension)
        
        
        self.first_level = DownBlock(in_chan, 16, time_dimension)
        self.second_level = DownBlock(16, 32, time_dimension)
        self.third_level = DownBlock(32, 64, time_dimension)
        self.fourth_level = DownBlock(64, 128, time_dimension)
        self.bottle_neck = DownBlock(128, 256, time_dimension)
        self.up_sample = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.fifth_level = UpBlock(256, 128, time_dimension)
        self.up_sample_5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) 
        self.sixth_level = UpBlock(128, 64, time_dimension)
        self.up_sample_6 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.seventh_level = UpBlock(64, 32, time_dimension)
        self.up_sample_7 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.eigth_level = UpBlock(32, 16, time_dimension)
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
        x5 = self.up_sample(x_bottle)
        
        x5 = torch.cat([x4_skip, x5], dim=1)
        x5 = self.fifth_level(x5, t)
        x6 = self.up_sample_5(x5)

        x6 = torch.cat([x3_skip, x6], dim = 1)
        x6 = self.sixth_level(x6, t) 
        x7 = self.up_sample_6(x6)

        x7 = torch.cat([x2_skip, x7], dim = 1)
        x7 = self.seventh_level(x7, t)
        x8 = self.up_sample_7(x7) 

        x8 = torch.cat([x1_skip, x8], dim = 1)
        x8 = self.eigth_level(x8, t) 
        
        output_x = self.last_conv(x8)
        
        return output_x 
