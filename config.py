# configuration file

from os import sep, path

# # 1:graysale 3:rgb
CHANNELS = 1
# laten vector dimemsion
LATENT_VECTOR_DIM = 256

# training params
# 4, 8, 16, 32, 64, 128, 256
BATCH_SIZE = (16, 16, 16, 16, 16, 8, 4)
FILTERS = (256, 256, 256, 256, 128, 64, 32)
EPOCHS = (25, 25, 25, 25, 25, 40, 50)

INPUT_IMAGE_FOLDER = path.join('D', sep, 'images', 'keras_png_slices_data')  # training data folder
# INPUT_IMAGE_FOLDER = path.join('D', sep, 'images', 'AKOA_Analysis')  # training data folder

# checkpoints settings
N_SAMPLES = 9       
# number of output images
OUTPUT_ROOT = path.join('D', sep, 'StyleGAN_out', 'output')
OUTPUT_IMAGE_FOLDER = path.join(OUTPUT_ROOT, 'images') # output image folder
OUTPUT_MODEL_FOLDER = path.join(OUTPUT_ROOT, 'models') # output model plot folder
OUTPUT_CKPTS_FOLDER = path.join(OUTPUT_ROOT, 'ckpts')  # check points folder
