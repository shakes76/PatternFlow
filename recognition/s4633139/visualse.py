#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021, H.WAKAYAMA, All rights reserved.
#  File: visualse.py
#  Author: Hideki WAKAYAMA
#  Contact: h.wakayama@uq.net.au
#  Platform: macOS Big Sur Ver 11.2.1, Pycharm pro 2021.1
#  Time: 19/10/2021, 17:30
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import matplotlib.pyplot as plt
import numpy as np


def dice_coef_vis(EPOCHS, TRAIN_COEFS, VAL_COEFS):
    """
    function for dice coefficient
    :param
        EPOCHS(array): epochs
        TRAIN_COEFS(array): train dice coefficients
        VAL_COEFS(array): validation dice coefficients
    :return
        plot with dice coefficients by epochs
    """
    X = np.arange(1, EPOCHS+1)
    plt.plot(X, TRAIN_COEFS, marker='.', markersize=10, label='train')
    plt.plot(X, VAL_COEFS, marker='.', markersize=10, label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('Dice coefficient')
    plt.xticks(X)
    plt.legend()
    plt.show()


def dice_loss_vis(EPOCHS, TRAIN_LOSS, VAL_LOSS):
    """
    function for dice loss
    :param
        EPOCHS(array): epochs
        TRAIN_LOSS(array): train dice losses
        VAL_LOSS(array): validation dice losses
    :return
        plot with dice loss by epochs
    """
    X = np.arange(1, EPOCHS+1)
    plt.plot(X, TRAIN_LOSS, marker='.', markersize=10, label='train')
    plt.plot(X, VAL_LOSS, marker='.', markersize=10, label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Loss')
    plt.xticks(X)
    plt.legend()
    plt.show()


def eval_dice_coef(target, pred_masks, idx):
  batch_size = len(pred_masks)
  somooth = 1.

  pred_flat = pred_masks.view(batch_size, -1)
  target_flat = target.view(batch_size, -1)

  intersection = (pred_flat*target_flat)
  dice_coef = (2.*intersection.sum(dim=1)+somooth)/(pred_flat.sum(dim=1)+target_flat.sum(dim=1)+somooth)
  return dice_coef[idx]


def segment_pred_mask(imgs, pred_masks, idx, alpha):
    """
    function to make a covered image with the predicted mask
    :param imgs(tensor[B,C,W,H]): 3 channels image
    :param pred_masks(tensor[B,C,W,H]): predicted mask
    :param idx(int): image index
    :param alpha(float): ratio for segmentation
    :return: segmentation image
    """
    seg_img = imgs[idx].clone()
    image_r = seg_img[0]
    image_r = image_r*(1-alpha*pred_masks[idx])+(pred_masks[idx]*pred_masks[idx]*alpha)
    segment_image = image_r.detach().squeeze()
    seg_img[0] = segment_image
    return seg_img


def plot_gallery(images, masks, pred_masks, n_row=5, n_col=4):
    """
    function to generate gallery
    :parameters
     images(tensor[B,C,W,H]): images
     masks(tensor[B,C,W,H]): target masks
     pred_masks(tensor[B,C,W,H]): predicted masks
     n_row: number of the row for the gallery
     n_col:  number of the column for the gallery
    :return
     gallery images
    """
    idxs = n_col * n_row
    plt.figure(figsize=(1.5 * n_col, 1.5 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.9, hspace=0.35)  # adjust layout parameters
    plt.suptitle('Segmentation', fontsize=15)

    for i in range(0, idxs, 4):
        # image
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title('image', fontsize=10)
        plt.axis('off')

        # target mask
        plt.subplot(n_row, n_col, i + 2)
        plt.imshow(masks[i].detach().squeeze(), cmap='gray')
        plt.title('target mask', fontsize=10)
        plt.axis('off')

        # predicted mask
        plt.subplot(n_row, n_col, i + 3)
        plt.imshow(pred_masks[i].detach().squeeze(), cmap='gray')
        plt.title('predicted mask', fontsize=10)
        plt.axis('off')

        # segmentation
        seg_img = segment_pred_mask(imgs=images, pred_masks=pred_masks, idx=i, alpha=0.5)
        plt.subplot(n_row, n_col, i + 4)
        plt.imshow(seg_img.permute(1, 2, 0))
        plt.title('dice_coef: {:.2f}'.format(eval_dice_coef(masks, pred_masks, i)), fontsize=10)
        plt.axis('off')
    plt.show()