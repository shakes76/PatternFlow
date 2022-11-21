import torch
import torch.nn as nn

# Hyper parameters for Improved 2D-Unet as specified in paper “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,”
# can adjust these to affect loss and dice coefficients of training and test of model. 
# These parameters achieve the target goal of >0.8 dice coefficient average on test set.
negativeSlope = 10**-2
pDrop = 0.3

class Improved2DUnet(nn.Module):
    """
    Implements a class object represent Improved 2D-UNet inheriting from nn.Module.
    """
    def __init__(self, in_channels = 3, out_channels = 1, features = [16, 32, 64, 128, 256]):
        """
        Defines and stores all layers that will be required to build model architecture.

        in_channels: number of channels for images in dataset.
        out_channels: number of desired channels for output of model.
        features: list describing output features of each context layer, ie: index 0 -> outputs features of context layer 1, index 1 -> outputs features of context layer 2
        """

        super(Improved2DUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.features_reversed = list(reversed(features))

        self.lrelu = nn.LeakyReLU(negative_slope=negativeSlope)
        self.dropout = nn.Dropout2d(p=pDrop)
        self.upScale = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        self.convs_context = nn.ModuleList()
        self.contexts = nn.ModuleList()
        self.norm_relus_context = nn.ModuleList()
        self.convs_norm_relu_local = nn.ModuleList()
        self.convs_local = nn.ModuleList()
        self.upSamples = nn.ModuleList()

        for i in range(5):
            if i == 0:
                self.convs_context.append(nn.Conv2d(self.in_channels, self.features[i], kernel_size=3, stride=1, padding=1, bias=False))
                self.convs_local.append(nn.Conv2d(self.features_reversed[i + 1], self.features_reversed[i + 1], kernel_size=1, stride=1, padding=0, bias=False))
            elif i == 4:
                self.convs_context.append(nn.Conv2d(self.features[i - 1], self.features[i], kernel_size=3, stride=2, padding=1, bias=False))
                self.convs_local.append(nn.Conv2d(self.features_reversed[i - 1], self.out_channels, kernel_size=1, stride=1, padding=0, bias=False))
            else:
                self.convs_context.append(nn.Conv2d(self.features[i - 1], self.features[i], kernel_size=3, stride=2, padding=1, bias=False))
                self.convs_local.append(nn.Conv2d(self.features_reversed[i - 1], self.features_reversed[i], kernel_size=1, stride=1, padding=0, bias=False))
            
            conv = self.norm_lrelu_conv(features[i], self.features[i])
            self.contexts.append(self.context(conv, conv))
            if i < 4:
                norm_lrelu = self.norm_lrelu(self.features[i])
                self.norm_relus_context.append(norm_lrelu)
        
        for p in range(4):
            self.convs_norm_relu_local.append(self.conv_norm_lrelu(self.features_reversed[p], self.features_reversed[p]))
            self.upSamples.append(self.up_sample(self.features_reversed[p], self.features_reversed[p + 1]))
        
        self.norm_local0 = nn.InstanceNorm2d(self.features_reversed[1])
        self.deep_segment_2_conv = nn.Conv2d(self.features_reversed[1], self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.deep_segment_3_conv = nn.Conv2d(self.features_reversed[2], self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def up_sample(self, feat_in, feat_out):
        """
        Up Sampling layer as described in Improved Unet Paper.

        feat_in: number of features into layer.
        feat_out: desired number of feature out of layer.
        """
        return nn.Sequential(
            nn.InstanceNorm2d(feat_in),
            self.lrelu,
            self.upScale,
            nn.Conv2d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(feat_out),
            self.lrelu
        )

    def context(self, conv1, conv2):
        """
        Contextualization layer as described in Improved Unet Paper.

        conv1: first 2D convolution of this context layer.
        conv2: second 2D convolution of this context layer.
        """
        return nn.Sequential(
            conv1,
            self.dropout,
            conv2
        )

    def norm_lrelu_conv(self, feat_in, feat_out):
        """
        2D convlution followed by instance 2D normalization followed by leaky relu activation with specificed negative slope in hyper parameters above.

        feat_in: number of features into layer.
        feat_out: desired number of feature out of layer.
        """
        return nn.Sequential(
			nn.Conv2d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm2d(feat_out),
			self.lrelu)

    def norm_lrelu(self, feat):
        """
        Instance 2D normalization followed by leaky relu activation with specificed negative slope in hyper parameters above.

        feat: number of features of current layer.
        """
        return nn.Sequential(
            nn.InstanceNorm2d(feat),
            self.lrelu
        )

    def conv_norm_lrelu(self, feat_in, feat_out):
        """
        2D convlution followed by instance 2D normalization followed by leaky relu activation with specificed negative slope in hyper parameters above.

        feat_in: number of features into layer.
        feat_out: desired number of feature out of layer.
        """
        return nn.Sequential(
			nn.Conv2d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm2d(feat_out),
			nn.LeakyReLU())
    
    def forward(self, x):
        """
        Outlines architecture of model for a forward pass, as described in Improved UNet Paper.
        Since this ISIC2017 2D-Unet outputs a binary segmention mask, torch.sigmoid is used as final layer instead of softmax.

        x: PyTorch Array
        return: PyTorch Array of same size as x, applying a forward pass on x.
        """
        residuals = dict()
        skips = dict()
        out = x
        
        #Contextualization level 1 to 5
        for i in range(5):
            out = self.convs_context[i](out)
            residuals[i] = out
            out = self.contexts[i](out)
            out += residuals[i]
            if (i < 4):
                out = self.norm_relus_context[i](out)
                skips[i] = out
            

        # localization level 1

        out = self.upSamples[0](out)
        out = self.convs_local[0](out)
        out = self.norm_local0(out)
        out = self.lrelu(out)
        
        # Localization level 2-5

        for j in range(4):
            out = torch.cat([out, skips[3-j]], dim=1)
            out = self.convs_norm_relu_local[j](out)
            if (j == 1):
                ds2 = out
            elif (j == 2):
                ds3 = out
            if (j == 3):
                out = self.convs_local[j+1](out)
            else:
                out = self.convs_local[j+1](out)
            if (j < 3):
                out = self.upSamples[j+1](out)

        
        #segment layer summation

        ds2_conv = self.deep_segment_2_conv(ds2)
        ds2_conv_upscale = self.upScale(ds2_conv)

        ds3_conv = self.deep_segment_3_conv(ds3)
        ds2_ds3_upscale = ds2_conv_upscale + ds3_conv

        ds2_ds3_upscale_upscale = self.upScale(ds2_ds3_upscale)
        out += ds2_ds3_upscale_upscale
    
        #Sigmoid Layer instead of softmax, justified in above docstring.

        out= torch.sigmoid(out)
        

        return out