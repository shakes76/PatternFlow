import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.io import imread, imshow


from tensorflow import keras
from tensorflow.keras import layers

import unet_model as mdl
import support_methods as sm

"""
Sources
"""


"""
All images are 3D MRi's of shape (256, 256, 128) in nibabel format (*.nii.gz).
Data and labels are in numpy arrays, float64.
MRi voxel values vary from 0.0 upwards.
The labels have 6 classes, labelled from 0.0 to 5.0.
"""

W = 256
D = 256
H = 128
CLASSES = 6


"""
Patients had from 1 to 8 MRI scans, a week apart. As scans for a given
patient are expected to be similar each patients scans have been considered as
one sample. All up there are 38 patients, and these have been distributed
between training, validation and testing at 27:7:4 with the number of images
at 158:35:18.
"""

# Data Sources Windows
# Data sources
X_TRAIN_DIR = 'D:\\prostate\\mr_train'
X_VALIDATE_DIR = 'D:\\prostate\\mr_validate'
X_TEST_DIR = 'D:\\prostate\\mr_test'
# Label sources
Y_TRAIN_DIR = 'D:\\prostate\\label_train'
Y_VALIDATE_DIR = 'D:\\prostate\\label_validate'
Y_TEST_DIR = 'D:\\prostate\\label_test'





# # Data sources Goliath
# # Data sources
# X_TRAIN_DIR = '~/prostate/mr_train'
# X_VALIDATE_DIR = '~/prostate/mr_validate'
# X_TEST_DIR = '~/prostate/mr_test'
# # Label sources
# Y_TRAIN_DIR = '~/prostate?label_train'
# Y_VALIDATE_DIR = '~/prostate/label_validate'
# Y_TEST_DIR = '~/prostate/label_test'

# Example data & label
img_mr = (nib.load(X_TRAIN_DIR + '\\Case_004_Week0_LFOV.nii.gz')).get_fdata()
img_label = (nib.load(Y_TRAIN_DIR + '\\Case_004_Week0_SEMANTIC_LFOV.nii.gz')).get_fdata()


image_train = [os.path.join(os.getcwd(), 'D:\\prostate\\mr_train', x)
               for x in os.listdir('D:\\prostate\\mr_train')]
image_validate = [os.path.join(os.getcwd(), 'D:\\prostate\\mr_validate', x)
                  for x in os.listdir('D:\\prostate\\mr_validate')]
image_test = [os.path.join(os.getcwd(), 'D:\\prostate\\mr_test', x)
          for x in os.listdir('D:\\prostate\\mr_test')]


label_train = [os.path.join(os.getcwd(),'D:\\prostate\\label_train', x)
               for x in os.listdir('D:\\prostate\\label_train')]
label_validate = [os.path.join(os.getcwd(),'D:\\prostate\\label_validate', x)
                  for x in os.listdir('D:\\prostate\\label_validate')]
label_test = [os.path.join(os.getcwd(),'D:\\prostate\\label_test', x)
              for x in os.listdir('D:\\prostate\\label_test')]





print(W)





def main():
    """ """
    # """ Checks dimensions of each image and label against expected."""
    # sm.dim_per_directory()

    # # Display raw data and label info
    # sm.data_info()
    #
    # # display images of data
    # sm.slices(img_mr)
    # # display images of labels
    # sm.slices(img_label)

    # # attempt to compile model    #update with BN, Relu
    # mdl.unet3d()



    # todo
    # files to laptop (git)
    # upsampling vs conc3DTranspose
    # check size of each image
    # find my original work
    # generator / sequence
    # normalise data, - mean / stdev  - tf.keras.utils.normalize(
    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/normalize
    # labels tf.one_hot( )  - tf.keras.utils.to_categorical
    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
    # sort / shuffle
    # model 3d
    # dsc
    # model.compile
    # model_checkpoint
    # model predict
    # model save / recover
    # plot predicted labels post
    # augmentation (distortion, slight rotations, horizontal flip
    #   translation (flip?), need to do same to label but siu's does that auto
    #   siu's github library
    #   see augmentation lib in lab sheets
    #   https://github.com/SiyuLiu0329/pyimgaug3d
    # cross validation
    # delete jupyter files from repo
    # save images to add to readme
    # customer 19, week 1 outsize, fix / resize / reshape...?

    # todo Issues non -critical
    # 1. Not printing images in subplots, works in jupyter
    # plot image slices & labels, pre - ensure access (try 3d later)
    # slices(img_mr)


if __name__ == '__main__':
    main()