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
        ## Upsampling Modules
        self.upsample5 = self.upsample(filter_size*16, filter_size*8)
        self.upsample4 = self.upsample(filter_size*8, filter_size*4)
        self.upsample3 = self.upsample(filter_size*4, filter_size*2)
        self.upsample2 = self.upsample(filter_size*2, filter_size)

        ## Localization Module
        self.localize4 = self.localization(filter_size*8)
        self.localize3 = self.localization(filter_size*4)
        self.localize2 = self.localization(filter_size*2)

        ## TODO: Segmentation Module
        self.segmentation = nn.Sequential()

        ## Softmax
        self.softmax = nn.Softmax2d()

        ## Leaky ReLU
        self.lrelu = nn.LeakyReLU(negative_slope=1e-2)

    def conv(self, filters_in, filters_out, stride, kernel_size=3, padding=1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                filters_in, 
                filters_out,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm2d(filters_out),
            self.lrelu
        )

    def context(self, filters) -> nn.Sequential:
        """
        Context module as described by Isensee et al.
        """
        return nn.Sequential(
            self.conv(filters, filters, stride=1),
            nn.Dropout2d(p=0.3),
            self.conv(filters, filters, stride=1),
        )
    
    def upsample(self, filters_in, filters_out) -> nn.Sequential:
        """ 
        Upsample module as described by Isensee et al.
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.conv(filters_in, filters_out, stride=1)
        )

    def localization(self, filters) -> nn.Sequential:
        """ 
        Localization module as described by Isensee et al.
        """
        return nn.Sequential(
            self.conv(filters, filters, 1),
            self.conv(filters, filters, 1, kernel_size=1, padding=0)
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
        ### TODO: refactor???
        # depth 1: 16 filters out
        out = self.conv1(x)
        residual = out
        out = self.context1(out)
        out = torch.add(residual, out) # elementwise sum
        out1 = out

        # depth 1: 32 filters out
        out = self.conv2(x)
        residual = out
        out = self.context2(out)
        out = torch.add(residual, out) # elementwise sum
        out2 = out

        # depth 3: 64 filters out
        out = self.conv3(x)
        residual = out
        out = self.context3(out)
        out = torch.add(residual, out) # elementwise sum
        out3 = out

        # depth 4: 128 filters out
        out = self.conv4(x)
        residual = out
        out = self.context4(out)
        out = torch.add(residual, out) # elementwise sum
        out4 = out

        # depth 5: 256 filters out
        out = self.conv5(x)
        residual = out
        out = self.context5(out)
        out = torch.add(residual, out) # elementwise sum
        out5 = out

        ### Localization Pathway
        # TODO

        return None
        
