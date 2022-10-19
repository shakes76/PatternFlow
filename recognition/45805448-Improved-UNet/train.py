import os
import tensorflow as tf
from keras.models import load_model, save_model
from dataset import load_image_dataset_from_directory, save_dataset
from modules import improved_unet

TRAINER = None

class Trainer:
    def __init__(self):
        self.image_size = None

        self.images_path = None
        self.masks_path = None

        self.datasets_path = None
        self.model_path = None

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.model = None

    def set_paths(self, images_path, masks_path, datasets_path, model_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.datasets_path = datasets_path
        self.model_path = model_path

    def load_data(self, train_split, valid_split, test_split, image_size=(1022, 767)):
        self.image_size = image_size

        images = load_image_dataset_from_directory(self.images_path, image_size=image_size)
        masks = load_image_dataset_from_directory(self.masks_path, image_size=image_size, color_mode='grayscale')

        num_samples = len(os.listdir(self.images_path))
        total = train_split + valid_split + test_split
        train_size = int(num_samples * (train_split / total))
        valid_size = int(num_samples * (valid_split / total))
        test_size = int(num_samples * (test_split / total))

        print(f'{num_samples} {train_size} {valid_size} {test_size}')

        train_images = images.take(train_size)
        valid_images = images.skip(train_size).take(valid_size)
        test_images = images.skip(train_size + valid_size).take(test_size)

        train_masks = masks.take(train_size)
        valid_masks = masks.skip(train_size).take(valid_size)
        test_masks = masks.skip(train_size + valid_size).take(test_size)

        print(tf.shape(list(train_images.as_numpy_iterator())))
        print(tf.shape(list(train_masks.as_numpy_iterator())))

        self.train_dataset = tf.data.Dataset.zip((train_images, train_masks))
        self.valid_dataset = tf.data.Dataset.zip((valid_images, valid_masks))
        self.test_dataset = tf.data.Dataset.zip((test_images, test_masks))

    def save_data(self):
        save_dataset(self.train_dataset, self.datasets_path + '/train')
        save_dataset(self.valid_dataset, self.datasets_path + '/valid')
        save_dataset(self.test_dataset, self.datasets_path + '/test')

    def build_model(self):
        self.model = improved_unet((self.image_size[0], self.image_size[1], 3), batch_size=3)

    def summarise_model(self):
        print(self.model.summary())

    def train_model(self, epochs=2):
        self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.valid_dataset)

    def load_model(self):
        self.model = load_model(self.model_path)

    def save_model(self):
        save_model(self.model, self.model_path, overwrite=True, include_optimizer=True)
    