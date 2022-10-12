## Loading and preprocessing ISIC dataset

import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np

from modules import img_path, CocoLikeDataset
from mrcnn.visualize import display_instances, display_top_masks
from mrcnn.utils import extract_bboxes

dataset_train = CocoLikeDataset()
dataset_train.load_data('/Users/wotah_man/Documents/UQ/Datasets/maskRCNN_Dataset/labels.json', '/Users/wotah_man/Documents/UQ/Datasets/maskRCNN_Dataset/train')
dataset_train.prepare()

#In this example, I do not have annotations for my validation data, so I am loading train data
dataset_val = CocoLikeDataset()
dataset_val.load_data('/Users/wotah_man/Documents/UQ/Datasets/maskRCNN_Dataset/labels.json', '/Users/wotah_man/Documents/UQ/Datasets/maskRCNN_Dataset/train')
dataset_val.prepare()


dataset = dataset_train
image_ids = dataset.image_ids
image_ids = np.random.choice(dataset.image_ids, 3)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    display_top_masks(image, mask, class_ids, dataset.class_names, limit=1)  #limit to total number of classes



# define image id
image_id = 0
# load the image
image = dataset_train.load_image(image_id)
# load the masks and the class ids
mask, class_ids = dataset_train.load_mask(image_id)

# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, dataset_train.class_names)


#Container
class data():
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

dataset = data(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test
)

#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(dataset.x_train[i])
#plt.show()