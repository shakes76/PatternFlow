# Any mention of "the paper" is referring to the Perceiver paper, i.e.
# https://arxiv.org/abs/2103.03206v2

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from perceiver_pytorch import Perceiver

class AdniClassifier(nn.Module):
    RESNET_OUTPUT_DIM = 2048
    PERCEIVER_NUM_LATENTS = 256		# paper appears to recommend 256 or 512
    PERCEIVER_LATENT_DIM = 512		# paper appears to recommend 512

    def __init__(self):
        super().__init__()

        # ResNet50 takes 3-channel images, but our data is 1-channel,
        # so we insert a 1-channel conv2d layer in front.
        # The hyper-params are copied from the conv1 layer in ResNet50
        self.pre_resnet = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        # Do transfer learning by using pre-trained weights from training on ImageNet
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Don't use the last fc layer. The resulting output is a vector of 2048 features.
        self.resnet = nn.Sequential(*tuple(resnet.children())[:-1])

        # Perceiver for classifying a sequence
        self.perceiver = Perceiver(
            num_freq_bands = 6,
            depth = 6,
            max_freq = 10.,
            input_channels = AdniClassifier.RESNET_OUTPUT_DIM,
            input_axis = 1,
            num_latents = AdniClassifier.PERCEIVER_NUM_LATENTS,
            latent_dim = AdniClassifier.PERCEIVER_LATENT_DIM,
            num_classes = 1,
        )

    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.reshape((B * S, C, H, W))
        out = self.pre_resnet(x)
        out = self.resnet(out)
        out = out.reshape((B, S, AdniClassifier.RESNET_OUTPUT_DIM))
        out = self.perceiver(out)
        return torch.sigmoid(out)

if __name__ == "__main__":
    device = torch.device("cpu")
    # ds = ADNI(device, "/home/jingweini/Documents/uni/COMP3710/report/data/AD_NC")
    # loader = DataLoader(ds, shuffle=True, batch_size=4)

    # seq, label = next(iter(loader))
    # print(seq.shape)
    # print(label.shape)

    # model = AdniClassifier().to(device)
    # out = model(seq)
    # print(out.shape)
