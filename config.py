# 1:graysale 3:rgb
CHANNELS = 1
# laten vector dimemsion
LATENT_VECTOR_DIM = 256


# training params
# 4, 8, 16, 32, 64, 128, 256
BATCH_SIZE = (16, 16, 16, 16, 16, 8, 4)
FILTERS = (256, 256, 256, 256, 128, 64, 32)
EPOCHS = (25, 25, 25, 25, 25, 40, 50)

# training data folder
INPUT_IMAGE_FOLDER = 'D:\images\keras_png_slices_data'
# INPUT_IMAGE_FOLDER = 'D:\images\AKOA_Analysis'

# checkpoints settings
N_SAMPLES = 9  # number of output samples
OUTPUT_IMAGE_FOLDER = 'output/images'      # output image folder
OUTPUT_MODEL_FOLDER = 'output/models'      # output model plot folder
OUTPUT_CKPTS_FOLDER = 'output/checkpoints'  # check points folder


########################
# mini batch for testing
# BATCH_SIZE = (3, 3)
# FILTERS = [256, 128]
# EPOCHS = (1, 1)
# INPUT_IMAGE_FOLDER = 'D:\minibatch'