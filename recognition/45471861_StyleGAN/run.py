# !/user/bin/env python
"""
The script trains the StyleGAN
"""

from train import Trainer

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

# Command Line Argument Method
# Local
# DATA = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_train"
# OUTPUT_DIR = "C:\\Users\\Zhien Zhang\\Desktop\\Other\\COMP3710\\StyleGAN\\recognition\\45471861_StyleGAN\\Output\\" \
#              "image_in_training"
# cloud
DATA = "/home/azureuser/cloudfiles/code/Users/zhien.zhang/keras_png_slices_data/keras_png_slices_train/keras_png_slices_train"
OUTPUT_DIR = "/home/azureuser/cloudfiles/code/Users/zhien.zhang/Output"

# 256
RESOLUTION = 256
G_INPUT_RES = 8
G_INIT_FILTERS = 512
D_FINAL_RES = 8
D_INPUT_FILTERS = 32
EPOCHS = 40
NEPTUNE = True
batch = 32
LATENT = 100
LR = 0.0001
FADE_IN_BASE = 80

# 128
# RESOLUTION = 128
# G_INPUT_RES = 8
# G_INIT_FILTERS = 512
# D_FINAL_RES = 8
# D_INPUT_FILTERS = 32
# EPOCHS = 30
# NEPTUNE = True
# batch = 64
# LATENT = 100

# 64
# RESOLUTION = 64
# G_INPUT_RES = 4
# G_INIT_FILTERS = 512
# D_FINAL_RES = 4
# D_INPUT_FILTERS = 64
# EPOCHS = 30
# NEPTUNE = True
# batch = 64

trainer = Trainer(DATA, OUTPUT_DIR, G_INPUT_RES, G_INIT_FILTERS, D_FINAL_RES, D_INPUT_FILTERS, FADE_IN_BASE,
                  batch=batch, resolution=RESOLUTION, epochs=EPOCHS, use_neptune=NEPTUNE)

trainer.train()
