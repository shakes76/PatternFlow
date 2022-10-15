## Loading and preprocessing annotated ISIC dataset

import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np

from modules import CocoLikeDataset
from mrcnn.visualize import display_instances, display_top_masks
from mrcnn.utils import extract_bboxes

dataset_train = CocoLikeDataset()
dataset_train.load_data('/Users/wotah_man/Documents/UQ/Datasets/maskRCNN_Dataset/labels.json', '/Users/wotah_man/Documents/UQ/Datasets/maskRCNN_Dataset/train')
dataset_train.prepare()

dataset_val = CocoLikeDataset()
dataset_val.load_data('/Users/wotah_man/Documents/UQ/Datasets/maskRCNN_Dataset/labels.json', '/Users/wotah_man/Documents/UQ/Datasets/maskRCNN_Dataset/train')
dataset_val.prepare()

'''
# display the training image
dataset = dataset_train
image_ids = dataset.image_ids
image_ids = np.random.choice(dataset.image_ids, 3)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    display_top_masks(image, mask, class_ids, dataset.class_names, limit=1)
'''

train = dataset_train

# display the training image with bounding box
image_id = 0
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)
bbox = extract_bboxes(mask)
display_instances(image, bbox, mask, class_ids, dataset_train.class_names)
