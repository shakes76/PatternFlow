import numpy as np
from matplotlib import pyplot as plt

from driver import *


def data_info():
    # data info
    img_mr = (nib.load(X_TRAIN_DIR + '\\Case_004_Week0_LFOV.nii.gz')).get_fdata()
    raw_data_info(img_mr)
    # label info
    img_label = (nib.load(Y_TRAIN_DIR + '\\Case_004_Week0_SEMANTIC_LFOV.nii.gz')).get_fdata()
    raw_data_info(img_label)


def raw_data_info(image):
    print("image information")
    print(type(image))
    print(image.dtype)
    print(image.shape)
    print(np.amin(image), np.amax(image))
    print()

def slices(img):
    slice_0 = img[127, :, :]
    slice_1 = img[:, 127, :]
    slice_2 = img[:, :, 63]
    show_slices([slice_0, slice_1, slice_2])

def show_slices(sliced):
    for i in sliced:
        plt.imshow(i.T)
        plt.show()

    # not working
    # fig, axes = plt.subplots(1, len(sliced))
    # for i, slice in enumerate(sliced):
    #     axes[i].imshow(slice.T, cmap="gray", origin="lower")

