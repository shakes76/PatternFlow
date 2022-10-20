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

        ## Leaky ReLU
        self.lrelu = nn.LeakyReLU(negative_slope=1e-2)

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
        self.localize4 = self.localization(filter_size*16, filter_size*8)
        self.localize3 = self.localization(filter_size*8, filter_size*4)
        self.localize2 = self.localization(filter_size*4, filter_size*2)

        ## 3x3 Convolution
        self.conv1_localization = self.conv(filter_size*2, filter_size*2, 1)

        ## Segmentation layers
        self.upscale = nn.Upsample(scale_factor=2)
        self.seg3 = self.conv(filter_size*4, filter_size, stride=1, kernel_size=1, padding=0)
        self.seg2 = self.conv(filter_size*2, filter_size, stride=1, kernel_size=1, padding=0)
        self.seg1 = self.conv(filter_size*2, filter_size, stride=1, kernel_size=1, padding=0)

        ## Output Layer
        self.out_layer = nn.Sequential(
            self.conv(filter_size, in_channels, stride=1)
        )


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

    def localization(self, filters_in, filters_out) -> nn.Sequential:
        """ 
        Localization module as described by Isensee et al.
        """
        return nn.Sequential(
            self.conv(filters_in, filters_in, 1),
            # 1x1 convolution to halve feature map
            self.conv(filters_in, filters_out, 1, kernel_size=1, padding=0)
        )
    
    def forward(self, x):
        """ 
        Generate predicted segmentation for given image
        NOTE: The usage of the notation for 'depth' here is nonstandard, 
                it is merely used for convenience
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
        out = self.conv2(out)
        residual = out
        out = self.context2(out)
        out = torch.add(residual, out) # elementwise sum
        out2 = out

        # depth 3: 64 filters out
        out = self.conv3(out)
        residual = out
        out = self.context3(out)
        out = torch.add(residual, out) # elementwise sum
        out3 = out

        # depth 4: 128 filters out
        out = self.conv4(out)
        residual = out
        out = self.context4(out)
        out = torch.add(residual, out) # elementwise sum
        out4 = out

        # depth 5: 256 filters out
        out = self.conv5(out)
        residual = out
        out = self.context5(out)
        out = torch.add(residual, out) # elementwise sum

        ### Localization Pathway
        # depth 5
        out = self.upsample5(out)

        # depth 4
        out = torch.cat((out, out4), dim=1)
        # conv1 256-256
        # conv2 256-128
        out = self.localize4(out)
        # upsample 128-64
        out = self.upsample4(out)

        # depth 3
        out = torch.cat((out, out3), dim=1)
        out = self.localize3(out)
        l3 = out # localization results to be used in segmentation layers
        out = self.upsample3(out) 

        # depth 2
        out = torch.cat((out, out2), dim=1)
        out = self.localize2(out)
        l2 = out
        out = self.upsample2(out)

        # depth 1
        out = torch.cat((out, out1), dim=1)
        out = self.conv1_localization(out)

        # segmentation layers
        s3 = self.seg3(l3)
        s3 = self.upscale(s3)
        s2 = self.seg2(l2)
        s23 = torch.add(s3, s2)
        s23 = self.upscale(s23)
        out = self.seg1(out)
        out = torch.add(s23, out)

        # output
        out = self.out_layer(out)
        return out
        

if __name__ == '__main__':
    x = torch.rand((2, 3, 256, 256)).to('cuda')
    iunet = IUNET(3, 16).to('cuda')
    y = iunet(x)
    print(x.shape)
    print(y.shape)
    