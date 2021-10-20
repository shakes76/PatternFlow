# !/user/bin/env python
"""
The script trains the StyleGAN
"""

from train import Trainer

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

# Command Line Argument Method
DATA = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_train"
OUTPUT_DIR = "C:\\Users\\Zhien Zhang\\Desktop\\Other\\COMP3710\\StyleGAN\\recognition\\45471861_StyleGAN\\Output\\" \
             "image_in_training"

trainer = Trainer(DATA, OUTPUT_DIR)

trainer.train()
