import tensorflow as tf
import os
import random
from skimage import io
import tensorflow_datasets as tfds

IMAGE_DIR = 'ISIC2018_Task1-2_Training_Input_x2/photos/'
MASK_DIR = 'ISIC2018_Task1_Training_GroundTruth_x2/photos/'
NUM_FILES = 2594

# TODO: Python doc

class ISICsDataLoader():

    def __init__(self, image_id_map, mask_id_map):
        self.image_id_map = image_id_map
        self.mask_id_map = mask_id_map
        self.image_ids = list(self.image_id_map.keys())

    def image_info(self, id):
        image = self.image_id_map[id]
        return image

    def load_image(self, id):
        return io.imread(self.image_id_map[id])

    def load_mask(self, id):
        return io.imread(self.mask_id_map[id])

def training_validation_ids(dir, valid_split):
    # For now, just load images all into CPU memory, should only be a few GB which should be fine
    # Then generate IDs using a dictionary mapping the ID to a filename
    training_image_id_map = {}
    validation_image_id_map = {}
    training_mask_id_map = {}
    validation_mask_id_map = {}
    ids = list(range(1, NUM_FILES + 1))
    random.seed(42)
    random.shuffle(ids)
    training_stop_id = round(NUM_FILES * (1 - valid_split))

    index = 0
    new_dir = dir + IMAGE_DIR
    ordered_file_names = os.listdir(new_dir)
    ordered_file_names.sort()
    for filename in ordered_file_names:
        if filename.endswith(".jpg"):
            id = ids[index]
            if id < training_stop_id:
                training_image_id_map[id] = filename
            else:
                validation_image_id_map[id] = filename
            index += 1

    index = 0
    new_dir = dir + MASK_DIR
    ordered_file_names = os.listdir(new_dir)
    ordered_file_names.sort()
    for filename in ordered_file_names:
        if filename.endswith(".png"):
            id = ids[index]
            if id < training_stop_id:
                training_mask_id_map[id] = filename
            else:
                validation_mask_id_map[id] = filename
            index += 1

    return training_image_id_map, training_mask_id_map, validation_image_id_map, validation_mask_id_map
