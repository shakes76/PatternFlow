# config file

import os


# model params
EPS = 1.e-8                                   # epsilon
CHANNELS = 1                                  # 1:graysale 3:rgb
LDIM = 128                                    # laten vector dimemsion
SRES = 4                                      # starting resolution
TRES = 128                                    # target resolution

# training params
BSIZE = (32, 32, 32, 32, 16, 8, 4)            # batch size of each resolution
FILTERS = (256, 256, 256, 256, 128, 64, 32)   # number of filters of each resolution
EPOCHS = (25, 25, 25, 25, 30, 35, 40)         # training epochs of each resolution
INPUT_IMAGE_FOLDER = 'TrainingImages'         # training image folder

# output params
NSAMPLES = 25                                 # number of output images
OUT_ROOT = 'output'                           # output root folder
IMAGE_DIR = os.path.join(OUT_ROOT, 'images')  # output image folder
MODEL_DIR = os.path.join(OUT_ROOT, 'models')  # output model plot folder
CKPTS_DIR = os.path.join(OUT_ROOT, 'ckpts')   # check points folder
LOG_DIR = os.path.join(OUT_ROOT, 'log')       # check points folder
