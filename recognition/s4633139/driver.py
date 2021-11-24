#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021, H.WAKAYAMA, All rights reserved.
#  File: driver.py
#  Author: Hideki WAKAYAMA
#  Contact: h.wakayama@uq.net.au
#  Platform: macOS Big Sur Ver 11.2.1, Pycharm pro 2021.1
#  Time: 19/10/2021, 17:30
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from dataloader import UNet_dataset
from model import IUNet
from model_train_val import model_train_val
from visualse import dice_coef_vis, segment_pred_mask, plot_gallery

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import torch.optim as optim

import matplotlib.pyplot as plt

def main():
    """execute model training and return dice coefficient plots"""

    #PARAMETERS
    FEATURE_SIZE=[16, 32, 64, 128]
    IN_CHANEL=3
    OUT_CHANEL=1

    IMG_TF = transforms.Compose([
                        transforms.Resize((FEATURE_SIZE[-1], FEATURE_SIZE[-1])),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
                        ])

    MASK_TF = transforms.Compose([
                        transforms.Resize((FEATURE_SIZE[-1],FEATURE_SIZE[-1])),
                        transforms.ToTensor(),
                        ])

    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #DATA PREPARATION
    dataset = UNet_dataset(img_transforms=IMG_TF, mask_transforms=MASK_TF)

    #shuffle index
    sample_size = len(dataset.imgs)
    train_size = int(sample_size * 0.5)
    split_size = sample_size - train_size

    val_size = split_size//2
    test_size = split_size - val_size

    #train, validation, test
    train_set, split_set = random_split(dataset, [train_size, split_size], generator=torch.Generator().manual_seed(123))
    val_set, test_set = random_split(split_set, [val_size, test_size], generator=torch.Generator().manual_seed(123))

    #data loader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    #MODEL
    model = IUNet(in_channels=IN_CHANEL, out_channels=OUT_CHANEL, feature_size=FEATURE_SIZE)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    #train,test
    TRAIN_DICE, VAL_DICE, VAL_LOSS, VAL_LOSS = model_train_val(model, optimizer, EPOCHS, train_loader, val_loader)

    #plot dice coefficient
    dice_coef_vis(EPOCHS, TRAIN_DICE, VAL_DICE)

    #segmentation
    for batch in test_loader:
        images, masks = batch
        break

    model.eval()
    pred_masks = model(images)

    plot_gallery(images, masks, pred_masks, n_row=5, n_col=4)

if __name__ == main():
    main()