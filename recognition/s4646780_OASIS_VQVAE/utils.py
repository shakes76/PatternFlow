import torch.nn as nn


def initialise_weights(model):
    """
    Function to initialize weights for GAN as specified in original GAN paper.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
