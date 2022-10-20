import torch, torchvision
import torch.nn as nn

class IUNET(nn.Module):
    """ 
    Pytorch implementation of the Improved UNET model
    Based on the paper: 
        "Brain Tumor Segmentation and Radiomics
        Survival Prediction:
        Contribution to the BRATS 2017 Challenge"
        Isensee et al. 2018 (https://arxiv.org/abs/1802.10508v1)
    """

    def __init__(self, in_channels, filter_size=16):
        """ 
        Initialize model modules
        in_channels: Number of expected channels in input image (3 for RGB)
        filter_size: 
        """
        super(IUNET, self).__init__()
        self.in_channels = in_channels
        self.filter_size = filter_size

        ### Context Pathway (encoder)
        ## Initial 3x3 Convolution
        self.initial_convolution = nn.Conv2d(
            self.in_channels, 
            self.filter_size, 
            kernel_size=3, stride=1, padding=1, bias=False
            )

        ## TODO: Context Module
        self.context_module = nn.Sequential()

        ## TODO: 3x3 stride 2 convolution
        self.context_conv = nn.Sequential()

        ### Localization Pathway (decoder)
        ## TODO: Upsampling Module
        self.upsampling = nn.Sequential()

        ## TODO: Localization Module
        self.localization = nn.Sequential()

        ## TODO: Segmentation Module
        self.segmentation = nn.Sequential()

        ## Softmax
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        """ 
        Generate predicted segmentation for given image
        x: image in the form of a tensor of shape (N, C, H, W)
            N: # of images in current batch
            C: # of channels
            H: image height
            W: image width
        returns segmentation for the image
        """
        # TODO:
        return None
        
