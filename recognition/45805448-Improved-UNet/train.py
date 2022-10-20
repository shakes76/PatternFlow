import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model, save_model
from dataset import load_image_dataset_from_directory, load_dataset, save_dataset, preprocess_dataset
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

        dataset = tf.data.Dataset.zip((images, masks))

        dataset = preprocess_dataset(dataset)

        num_samples = len(os.listdir(self.images_path))
        total = train_split + valid_split + test_split
        train_size = int(num_samples * (train_split / total))
        valid_size = int(num_samples * (valid_split / total))
        test_size = int(num_samples * (test_split / total))

        self.train_dataset = dataset.take(train_size)
        self.valid_dataset = dataset.skip(train_size).take(valid_size)
        self.test_dataset = dataset.skip(train_size + valid_size).take(test_size)

        for element in self.train_dataset:
            print(element)
            break

    def load_existing_data(self):
        self.train_dataset = load_dataset(self.datasets_path + '/train')
        self.valid_dataset = load_dataset(self.datasets_path + '/valid')
        self.test_dataset = load_dataset(self.datasets_path + '/test')

    def save_data(self):
        save_dataset(self.train_dataset, self.datasets_path + '/train')
        save_dataset(self.valid_dataset, self.datasets_path + '/valid')
        save_dataset(self.test_dataset, self.datasets_path + '/test')

    def display_samples_from_data(self):
        # batch = tf.Tensor(list(self.train_dataset.take(1).as_numpy_iterator()), dtype=tf.float32)
        print(self.train_dataset.cardinality().numpy())
        for element in self.train_dataset:
            train_images, train_masks = element[0], element[1]
            print(tf.shape(train_images))
            print(tf.shape(train_masks))

            plt.figure(figsize=(10,10))
            for i in range(6):
                plt.subplot(6, 2, i*2 + 1)
                plt.imshow(train_images[i])
                plt.axis('off')
                plt.title('Image')

                plt.subplot(6, 2, i*2 + 2)
                plt.imshow(train_masks[i], vmin=0, vmax=1)
                plt.axis('off')
                plt.title('Actual Mask')

            plt.savefig('testdisplaysamples.png')
            break

    def build_model(self):
        self.model = improved_unet((self.image_size[0], self.image_size[1], 3), batch_size=6)

    def summarise_model(self):
        print(self.model.summary())

    def train_model(self, epochs=2):
        self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.valid_dataset, verbose=2)

    def load_model(self):
        self.model = load_model(self.model_path)

    def save_model(self):
        save_model(self.model, self.model_path, overwrite=True, include_optimizer=True)
    