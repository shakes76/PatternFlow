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
        self.conv1 = self.conv(in_channels, filter_size, 1)

        ## Context Modules
        self.context1 = self.context(filter_size)
        self.context2 = self.context(filter_size*2)
        self.context3 = self.context(filter_size*4)
        self.context4 = self.context(filter_size*8)
        self.context5 = self.context(filter_size*16)

        ## 3x3 stride 2 convolutions
        self.conv2 = self.conv(filter_size, filter_size*2, 2)
        self.conv3 = self.conv(filter_size*2, filter_size*4, 2)
        self.conv4 = self.conv(filter_size*4, filter_size*8, 2)
        self.conv5 = self.conv(filter_size*8, filter_size*16, 2)

        ### Localization Pathway (decoder)
        ## TODO: Upsampling Module
        self.upsampling = nn.Sequential()

        ## TODO: Localization Module
        self.localization = nn.Sequential()

        ## TODO: Segmentation Module
        self.segmentation = nn.Sequential()

        ## Softmax
        self.softmax = nn.Softmax2d()

        ## Leaky ReLU
        self.lrelu = nn.LeakyReLU(negative_slope=1e-2)

    def conv(self, filters_in, filters_out, stride):
        return nn.Sequential(
            nn.Conv2d(
                filters_in, 
                filters_out,
                kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(filters_out),
            self.lrelu
        )

    def context(self, filters):
        """
        Context module as described by Isensee et al.
        """
        return nn.Sequential(
            nn.Conv2d(filters, filters, bias=False),
            nn.InstanceNorm2d(filters),
            self.lrelu,
            nn.Dropout2d(p=0.3),
            nn.Conv2d(filters, filters, bias=False),
            nn.InstanceNorm2d(filters),
            self.lrelu
        )

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
        ### Context Pathway
        return None
        
