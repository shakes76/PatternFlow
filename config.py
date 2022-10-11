# configuration file

from os import sep, path


CHANNELS = 1              # 1:graysale 3:rgb
LATENT_VECTOR_DIM = 128   # laten vector dimemsion

# training params
SRES = 4                                                     # starting resolution
TRES = 256                                                   # target resolution
BATCH_SIZE = (16, 16, 16, 16, 16, 8, 4)                      # batch size of each resolution
FILTERS = (256, 256, 256, 256, 128, 64, 32)                  # number of filters of each resolution
EPOCHS = (25, 25, 25, 25, 25, 40, 50)                        # training epochs of each resolution

INPUT_IMAGE_FOLDER = path.join('D:', sep, 'images', 'keras_png_slices_data')  # training data folder

# checkpoints settings
N_SAMPLES = 9                                                  # number of output images

# -root
#   |-ckpts
#   |-images
#   |-models
OUTPUT_ROOT = path.join('D:', sep, 'StyleGAN_out', 'output')   # root of output folder
OUTPUT_IMAGE_FOLDER = path.join(OUTPUT_ROOT, 'images')         # output image folder
OUTPUT_MODEL_FOLDER = path.join(OUTPUT_ROOT, 'models')         # output model plot folder
OUTPUT_CKPTS_FOLDER = path.join(OUTPUT_ROOT, 'ckpts')          # check points folder

# epsilon
EPS = 1.e-8
