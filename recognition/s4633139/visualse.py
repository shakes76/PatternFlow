import matplotlib.pyplot as plt
import numpy as np

def dice_coef_vis(EPOCHS, TRAIN_COEFS, TEST_COEFS):
    X = np.arange(1, EPOCHS+1)
    plt.plot(X, TRAIN_COEFS, marker='.', markersize=10, label='train')
    plt.plot(X, TEST_COEFS, marker='.', markersize=10, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Dice coefficient')
    plt.xticks(X)
    plt.legend()
    plt.show()


def dice_loss_vis(EPOCHS, TRAIN_LOSS, TEST_LOSS):
    X = np.arange(1, EPOCHS+1)
    plt.plot(X, TRAIN_LOSS, marker='.', markersize=10, label='train')
    plt.plot(X, TEST_LOSS, marker='.', markersize=10, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Loss')
    plt.xticks(X)
    plt.legend()
    plt.show()


def pred_mask(img, pred_mask, alpha=5):
    seg_img = img.clone()
    image_r = seg_img[0]
    image_r = image_r*(1-alpha*pred_mask)+(pred_mask*pred_mask*alpha)
    segmentation = image_r.detach().squeeze()
    seg_img[0] = segmentation
    plt.imshow(seg_img.permute(1,2,0))
    plt.show()


def segment_pred_mask(img, pred_mask, alpha=0.5):
    seg_img = img.clone()
    image_r = seg_img[0]
    image_r = image_r*(1-alpha*pred_mask)+(pred_mask*pred_mask*alpha)
    segment_img_r = image_r.detach().squeeze()
    seg_img[0] = segment_img_r
    plt.imshow(seg_img.permute(1,2,0))
    plt.show()