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
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        self.convs_context = list()
        for i in range(4):
            if i == 0:
                self.convs_context[i] = nn.Conv2d(self.in_channels, self.features[i], kernel_size=3, stride=1, padding=1, bias=False)
            self.convs_context[i + 1] = nn.Conv2d(self.features[i], self.features[i + 1], kernel_size=3, stride=1, padding=1, bias=False)
        
        self.contexts = list()
        self.norm_relus_context = list()
        for j in range(5):
            conv = self.norm_lrelu_conv(features[j], self.features[j])
            self.contexts[j] = self.context(conv, conv)
            if j < 4:
                norm_lrelu = self.norm_lrelu(self.features[j])
                self.norm_relus_context[j] = norm_lrelu
        
        self.convs_local = list()
        for k in range(5):
            if k == 0:
                self.convs_local[k] = nn.Conv2d(self.features_reversed[k + 1], self.features_reversed[k + 1], kernel_size=1, stride=1, padding=0, bias=False)
            else:
                self.convs_local[k] = nn.Conv2d(self.features_reversed[k - 1], self.features_reversed[k], kernel_size=1, stride=1, padding=0, bias=False)

        self.deep_segment_2_conv = nn.Conv2d(self.features_reversed[1], self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.deep_segment_3_conv = nn.Conv2d(self.features_reversed[2], self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def up_sample(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm2d(feat_in),
            self.lrelu,
            self.up_sample,
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
    
    def forward(self, x):

        return x