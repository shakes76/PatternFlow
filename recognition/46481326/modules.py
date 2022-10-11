# %%
"""Import libraries required for PyTorch"""
import torch # Import PyTorch
import torch.nn as nn # Import PyTorch Neural Network

# %%
class Hyperparameters():
    def get_rate_learn(self): return 2e-4 # Rate of learn of the optimizer
    def get_size_batch(self): return 128 # Size of batches for the PyTorch Dataloader(s)
    def get_size_image(self): return 64
    def get_channels_image(self): return 1 # Number of input channels (from the image)
    def get_size_z(self): return 100 # Size of the latent space
    def get_num_epoch(self): return 20 # Number of training epoch(s)
    def get_fn_loss(self): return nn.BCELoss() # Defines the loss function to be Binary Cross Entropy (BCE)
    
# %%
class UNetEncode(nn.Module):
    """UNet class for encoding component of UNet

    Args:
        nn (Module): nn (Module): Abstract class for PyTorch neural networks.
    """
    def __init__(self, channels_in, channels_out):
        """Constructor for UNetEncode

        Args:
            channels_in (int): Number of channel into the model
            channels_out (int): Number of channels output from the model
        """
        super(UNetEncode, self).__init__() # Initiate abstract class
        self.layer1 = nn.Sequential(nn.Conv2d(channels_in, 64, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(512, 1024, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(1024, 1024, 3, stride=1, padding=1), nn.ReLU())
        self.maxpool = nn.MaxPool2d(2, stride=2)
        
    def forward(self, x_layer5, x_layer4, x_layer3, x_layer2, x_layer1):
        """Takes in a time step of the network and returns the next time step.

        Args:
            x (_type_): The current time step of the network.
        """
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(self.maxpool(x_layer1))
        x_layer3 = self.layer3(self.maxpool(x_layer2))
        x_layer4 = self.layer4(self.maxpool(x_layer3))
        x_layer5 = self.layer5(self.maxpool(x_layer4))
        
        return (x_layer5, x_layer4, x_layer3, x_layer2, x_layer1)
# %%
class UNetDecode(nn.Module):
    """UNet class for encoding component of UNet

    Args:
        nn (Module): nn (Module): Abstract class for PyTorch neural networks.
    """
    def __init__(self, channels_in, channels_out):
        """Constructor for UNetEncode

        Args:
            channels_in (int): Number of channel into the model
            channels_out (int): Number of channels output from the model
        """
        super(UNetDecode, self).__init__() # Initiate abstract class
        self.layer1 = nn.Conv2d(64, 2, 3, stride=1, padding=1)
        self.layer2 = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(256, 128, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(512, 256, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(1024, 512, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.ReLU())
        self.deconv_5_4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.deconv_4_3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.deconv_3_2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv_2_1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        
    def forward(self, x_layer5, x_layer4, x_layer3, x_layer2, x_layer1):
        """Takes in a time step of the network and returns the next time step.

        Args:
            x (_type_): The current time step of the network.
        """
        x = self.deconv_5_4(x_layer5)
        x_layer4 = torch.cat([x, x_layer4], dim=1)
        x_layer4 = self.layer5(x_layer4)

        x = self.deconv_4_3(x_layer4)
        x_layer3 = torch.cat([x, x_layer3], dim=1)
        x_layer3 = self.layer4(x_layer3)

        x = self.deconv_3_2(x_layer3)
        x_layer2 = torch.cat([x, x_layer2], dim=1)
        x_layer2 = self.layer3(x_layer2)

        x = self.deconv_2_1(x_layer2)
        x_layer1 = torch.cat([x, x_layer1], dim=1)
        x_layer1 = self.layer2(x_layer1)

        x = self.layer1(x_layer1)
		
        return x
# %%
class UNet():
    """UNet class for image denoising

    Args:
        nn (Module): Abstract class for PyTorch neural networks.
    """