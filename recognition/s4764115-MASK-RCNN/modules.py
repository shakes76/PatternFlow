## Components for the network

import tensorflow as tf
import numpy as np
import utils
import keras
import os
import time
import json

from mrcnn import model as modellib, utils
from PIL import Image, ImageDraw

# Classes
class path:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test

img_path = path(
    '/Users/wotah_man/Documents/UQ/ISIC Dataset/Train',
    '/Users/wotah_man/Documents/UQ/ISIC Dataset/Val',
    '/Users/wotah_man/Documents/UQ/ISIC Dataset/Test'
)

# Loading data
class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
                
    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids


# Functions
def normalize(dataset):
    '''normalize the image data to 0~1 float'''
    for img, lbl in dataset:
        x = tf.math.divide(img, 255.0)
        y = lbl
    return x, y

def backbone():
    '''
    the backbone model
    (in this case it would be a classifier from my previous assignment)
    note that we only need the features from this, not the final class
    '''

    img_size = 512
    num_channels = 1

    input = tf.keras.Input(shape=(img_size, img_size, num_channels))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='LeakyReLU', padding='same')(input)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='LeakyReLU', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='LeakyReLU', padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='LeakyReLU', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='LeakyReLU', padding='same')(x)
    featuremap = tf.keras.layers.Conv2D(128, (3, 3), activation='LeakyReLU', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(featuremap)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2048, activation='LeakyReLU')(x)
    x = tf.keras.layers.Dense(1024, activation='LeakyReLU')(x)
    x = tf.keras.layers.Dense(512, activation='LeakyReLU')(x)
    x = tf.keras.layers.Dense(256, activation='LeakyReLU')(x)
    x = tf.keras.layers.Dense(64, activation='LeakyReLU')(x)
    output = tf.keras.layers.Dense(32, activation='LeakyReLU')(x)

    backbone = tf.keras.models.Model(input, output, name='backbone')

    return backbone

