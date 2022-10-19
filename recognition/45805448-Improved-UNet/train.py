import os
import tensorflow as tf
from dataset import load_image_dataset_from_directory
from modules import improved_unet

TRAINER = None

class Trainer:
    def __init__(self):
        self.images_path = None
        self.masks_path = None

        self.datasets_path = None
        self.model_path = None

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.model = None

def set_paths(self, images_path, masks_path):
    self.images_path = images_path
    self.masks_path = masks_path

def load_data(self, train_split, valid_split, test_split, image_size=(1022, 767)):
    images = load_image_dataset_from_directory(self.images_path, image_size=image_size)
    masks = load_image_dataset_from_directory(self.masks_path, image_size=image_size)

    num_samples = len(os.listdir(self.images_path))
    train_size = int(num_samples * train_split)
    valid_size = int(num_samples * valid_split)
    test_size = int(num_samples * test_split)

    train_images = images.take(train_size)
    valid_images = images.skip(train_size).take(valid_size)
    test_images = images.skip(train_size + valid_size).take(test_size)

    train_masks = masks.take(train_size)
    valid_masks = masks.skip(train_size).take(valid_size)
    test_masks = masks.skip(train_size + valid_size).take(test_size)

    self.train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks)).batch(2)
    self.valid_dataset = tf.data.Dataset.from_tensor_slices((valid_images, valid_masks)).batch(2)
    self.test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks)).batch(2)

def build_model(self):
    train_images = self.train_dataset.take(1)
    num_batches, batch_size, height, width, num_channels = train_images.shape

    self.model = improved_unet((height, width, num_channels), batch_size=batch_size)

def train_model(self, epochs=2):
    self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.valid_dataset)