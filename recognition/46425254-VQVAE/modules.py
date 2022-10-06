import torch
import torch.nn as nn
import torch.utils as utils
import torchvision


#Setting Global Parameters
image_dim = (1,256,256)
learning_rate = 0.0001
latent_space = 256

class Encoder(nn.Module):
    
    def __init__(self):
        #3 convolutional layers for a latent space of 64
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(64, 128, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(128, latent_space, kernel_size=4, stride = 2, padding = 1),
            nn.Tanh(),)
        
    
    def forward(self, x):
        return self.model(x)
            
            

    
    
class Decoder(nn.Module):
    
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(latent_space, 128, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(128, 64, kernel_size= 4, stride = 2, padding = 1),
            nn.BatchNormal(64),
            nn.LeakyReLU(0.1),
            
            nn.Conv(64, 1, kernel_size = 4, stride = 2, padding = 1)
            )
        
        
    def forward(self, x):
        return self.model(x)
    
    