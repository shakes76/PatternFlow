##Training the model

import os
import sys
import numpy as np

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from dataset import train

# configuration for the model
class LesionConfig(Config):
	NAME = 'lesion_cfg_coco'
	NUM_CLASSES = 1 + 1
	STEPS_PER_EPOCH = 100
config = LesionConfig()
config.display() 

ROOT_DIR = os.path.abspath('./')
# Import Mask RCNN
sys.path.append(ROOT_DIR)
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, '/Users/wotah_man/Documents/Github/mask_rcnn_coco.h5')

# define and train the model
model = MaskRCNN(mode='training', model_dir=DEFAULT_LOGS_DIR, config=config)
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc',  'mrcnn_bbox', 'mrcnn_mask'])
model.train(train, train, learning_rate=config.LEARNING_RATE, epochs=25, layers='heads')
