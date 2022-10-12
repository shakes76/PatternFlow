# config file

import os


# model params
EPS = 1.e-8                                   # epsilon
CHANNELS = 1                                  # 1:graysale 3:rgb
LDIM = 128                                    # laten vector dimemsion
SRES = 4                                      # starting resolution
TRES = 256                                    # target resolution

# training params
BSIZE = (16, 16, 16, 16, 16, 8, 4)            # batch size of each resolution
FILTERS = (256, 256, 256, 256, 128, 64, 32)   # number of filters of each resolution
EPOCHS = (25, 25, 25, 25, 25, 35, 50)         # training epochs of each resolution
INPUT_IMAGE_FOLDER = 'keras_png_slices_data'  # training image folder

# output params
NSAMPLES = 9                                  # number of output images
OUT_ROOT = 'output'                           # output root folder
IMAGE_DIR = os.path.join(OUT_ROOT, 'images')  # output image folder
MODEL_DIR = os.path.join(OUT_ROOT, 'models')  # output model plot folder
CKPTS_DIR = os.path.join(OUT_ROOT, 'ckpts')   # check points folder
