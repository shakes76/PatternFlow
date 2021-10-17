from IUNet_dataloader import UNet_dataset
from ImprovedUNet import IUNet
from IUNet_train_test import model_train_test
from visualse import dice_coef_vis, segment_pred_mask

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import torch.optim as optim


def main():
    """
    execute model training and return dice coefficient plots
    """

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
    EPOCHS = 15
    LR = 0.001

    #DATA PREPARATION
    dataset = UNet_dataset(img_transforms=IMG_TF, mask_transforms=MASK_TF)

    #shuffle index
    sample_size = len(dataset.imgs)
    train_size = int(sample_size * 0.8)
    test_size = sample_size - train_size

    #train and test set
    train_set, test_set = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(123))

    #data loader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    #MODEL
    model = IUNet(in_channels=IN_CHANEL, out_channels=OUT_CHANEL, feature_size=FEATURE_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    #train,test
    TRAIN_DICE, TEST_DICE, TRAIN_LOSS, TEST_LOSS = model_train_test(model, optimizer, EPOCHS, train_loader, test_loader)

    #plot dice coefficient
    dice_coef_vis(EPOCHS, TRAIN_DICE, TEST_DICE)

    #segmentation
    for batch in train_loader:
        x, y = batch
        break

    img = x[0]
    model.eval()
    pred_mask = model(x)[0]
    segment_pred_mask(img, pred_mask, alpha=0.5)

if __name__ == main():
    main()



