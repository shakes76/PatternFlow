##Training the model

import os
import sys
import numpy as np

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from dataset import train
from modules import LesionConfig

config = LesionConfig()
config.display() 

ROOT_DIR = os.path.abspath('./')
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, '/Users/wotah_man/Documents/Github/mask_rcnn_coco.h5')

# define and train the model
model = MaskRCNN(mode='training', model_dir=DEFAULT_LOGS_DIR, config=config)
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc',  'mrcnn_bbox', 'mrcnn_mask'])

Trained_model = model.train(train, train, learning_rate=config.LEARNING_RATE, epochs=25, layers='heads')
