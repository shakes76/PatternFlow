import torch
import torch.nn as nn

negativeSlope = 10^-2
pDrop = 0.3

class Improved2DUnet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, features = [16, 32, 64, 128, 256]):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.features_reversed = list(reversed(features))

        self.lrelu = nn.LeakyReLU(negative_slope=negativeSlope)
        self.dropout = nn.Dropout2d(p=pDrop)
        self.upScale = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        self.convs_context = list()
        for i in range(5):
            if i == 0:
                self.convs_context[i] = nn.Conv2d(self.in_channels, self.features[i], kernel_size=3, stride=1, padding=1, bias=False)
            else:
                self.convs_context[i] = nn.Conv2d(self.features[i - 1], self.features[i], kernel_size=3, stride=1, padding=1, bias=False)
        
        self.contexts = list()
        self.norm_relus_context = list()
        for j in range(5):
            conv = self.norm_lrelu_conv(features[j], self.features[j])
            self.contexts[j] = self.context(conv, conv)
            if j < 4:
                norm_lrelu = self.norm_lrelu(self.features[j])
                self.norm_relus_context[j] = norm_lrelu
        
        self.convs_norm_relu_local = list()
        for p in range(4):
            self.convs_norm_relu_local[p] = self.conv_norm_lrelu(self.features_reversed[p], self.features_reversed[p])

        self.convs_local = list()
        for k in range(5):
            if k == 0:
                self.convs_local[k] = nn.Conv2d(self.features_reversed[k + 1], self.features_reversed[k + 1], kernel_size=1, stride=1, padding=0, bias=False)
            else:
                self.convs_local[k] = nn.Conv2d(self.features_reversed[k - 1], self.features_reversed[k], kernel_size=1, stride=1, padding=0, bias=False)
        
        self.upSamples = list()
        for l in range(4):
            self.upSamples[l] = self.up_sample(self.features_reversed[l], self.features_reversed[l + 1])
        
        self.norm_local0 = nn.InstanceNorm2d(self.features_reversed[1])

        self.deep_segment_2_conv = nn.Conv2d(self.features_reversed[1], self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.deep_segment_3_conv = nn.Conv2d(self.features_reversed[2], self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def up_sample(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm2d(feat_in),
            self.lrelu,
            self.upScale,
            nn.Conv2d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(feat_out),
            self.lrelu
        )

    def context(self, conv1, conv2):
        return nn.Sequential(
            conv1,
            self.dropout,
            conv2
        )

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
			nn.Conv2d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm2d(feat_out),
			self.lrelu)

    def norm_lrelu(self, feat):
        return nn.Sequential(
            nn.InstanceNorm2d(feat),
            self.lrelu
        )

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
			nn.Conv2d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm2d(feat_out),
			nn.LeakyReLU())
    
    def forward(self, x):
        residuals = list()
        skips = list()

        #Context level 1 to 5
        for i in range(5):
            out = self.convs_context[i](x)
            residuals[i] = out
            out = self.contexts[i](out)
            out += residuals[i]
            if (i < 4):
                out = self.norm_relus_context[i](out)
                skips[i] = out

        #level 0 localization

        out = self.upSamples[0](out)
        out = self.convs_local[0](out)
        out = self.norm_local0(out)
        out = self.lrelu(out)

        # Local level 1-4

        for j in range(4):
            out = torch.cat([out, skips[3-j]], dim=1)
            out = self.convs_norm_relu_local[j](out)
            if (j == 1):
                ds2 = out
            elif (j == 2):
                ds3 = out
            if (j == 3):
                out_pred = self.convs_local[j+1](out)
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

        out = out_pred + ds2_ds3_upscale_upscale
        seg_layer = out
        out = out.permute(0,2,3,1).contiguous().view(-1, self.out_channels)
        out = self.softmax(out)

        return out, seg_layer