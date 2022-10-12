# configuration file

import os

# epsilon
EPS = 1.e-8

CHANNELS = 1              # 1:graysale 3:rgb
LDIM = 128   # laten vector dimemsion

# training params
SRES = 4                                      # starting resolution
TRES = 256                                    # target resolution
BSIZE = (16, 16, 16, 16, 16, 8, 4)            # batch size of each resolution
FILTERS = (256, 256, 256, 256, 128, 64, 32)   # number of filters of each resolution
EPOCHS = (25, 25, 25, 25, 25, 35, 50)         # training epochs of each resolution

# INPUT_IMAGE_FOLDER = os.path.join('D:', os.sep, 'images', 'keras_png_slices_data')
INPUT_IMAGE_FOLDER = os.path.join('D:', os.sep, 'images', 'minibatch')
# INPUT_IMAGE_FOLDER = os.path.join('D:', os.sep, 'images', 'ADNI_AD_NC_2D')
# INPUT_IMAGE_FOLDER = path.join('D:', sep, 'images', 'AKOA_Analysis')  


NSAMPLES = 9      # number of output images

# -root
#   |-ckpts
#   |-images
#   |-models
OUTPUT_ROOT = os.path.join('D:', os.sep, 'test_out')   
IMAGE_DIR = os.path.join(OUTPUT_ROOT, 'images')         # output image folder
MODEL_DIR = os.path.join(OUTPUT_ROOT, 'models')         # output model plot folder
CKPTS_DIR = os.path.join(OUTPUT_ROOT, 'ckpts')          # check points folder
