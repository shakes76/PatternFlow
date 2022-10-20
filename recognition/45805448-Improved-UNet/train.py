import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model, save_model
from dataset import load_image_dataset_from_directory, load_dataset, save_dataset, preprocess_dataset
from modules import improved_unet

TRAINER = None
BATCH_SIZE = 5

class Trainer:
    def __init__(self):
        self.batch_size = None
        self.image_height = None
        self.image_width = None

        self.images_path = None
        self.masks_path = None

        self.datasets_path = None
        self.model_path = None

        self.full_dataset = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.model = None

    def set_paths(self, images_path, masks_path, dataset_path, model_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.dataset_path = dataset_path
        self.model_path = model_path

    def load_data(self, image_size=(1022, 767)):
        images = load_image_dataset_from_directory(self.images_path, image_size=image_size, batch_size=BATCH_SIZE)
        masks = load_image_dataset_from_directory(self.masks_path, image_size=image_size, batch_size=BATCH_SIZE, color_mode='grayscale')

        self.full_dataset = tf.data.Dataset.zip((images, masks))
        self.full_dataset = preprocess_dataset(self.full_dataset)

    def load_existing_data(self):
        self.full_dataset = load_dataset(self.dataset_path)

    def save_data(self):
        save_dataset(self.full_dataset, self.dataset_path)

    def split_data(self, train_split, valid_split, test_split):
        num_samples = len(os.listdir(self.images_path))
        total = train_split + valid_split + test_split
        train_size = int(num_samples * (train_split / total) / BATCH_SIZE / 2)
        valid_size = int(num_samples * (valid_split / total) / BATCH_SIZE / 2)
        test_size = int(num_samples * (test_split / total) / BATCH_SIZE / 2)

        self.train_dataset = self.full_dataset.take(train_size)
        self.valid_dataset = self.full_dataset.skip(train_size).take(valid_size)
        self.test_dataset = self.full_dataset.skip(train_size + valid_size).take(test_size)

        for test_batch in self.test_dataset:
            test_images, test_masks = test_batch[0], test_batch[1]
            self.batch_size, self.image_height, self.image_width, _ = tf.shape(test_images)

            print(f'Shape of input: {tf.shape(test_images)}')
            print(f'Shape of output: {tf.shape(test_masks)}')
            break

        print(f'Training-Validation-Testing Batch Split: ' +
                f'{self.train_dataset.cardinality().numpy()}-' +
                f'{self.valid_dataset.cardinality().numpy()}-' +
                f'{self.test_dataset.cardinality().numpy()}\n' +
                f'Batch size: {self.batch_size}')

    def display_data(self):
        # batch = tf.Tensor(list(self.train_dataset.take(1).as_numpy_iterator()), dtype=tf.float32)
        for element in self.train_dataset:
            train_images, train_masks = element[0], element[1]

            plt.figure(figsize=(10,10))
            for i in range(BATCH_SIZE):
                plt.subplot(BATCH_SIZE, 2, i*2 + 1)
                plt.imshow(train_images[i])
                plt.axis('off')
                plt.title('Image')

                plt.subplot(BATCH_SIZE, 2, i*2 + 2)
                plt.imshow(train_masks[i], vmin=0, vmax=1)
                plt.axis('off')
                plt.title('Actual Mask')

            plt.savefig('pretraining_samples.png')
            break

    def build_model(self):
        self.model = improved_unet((self.image_height, self.image_width, 3), batch_size=self.batch_size)

    def summarise_model(self):
        print(self.model.summary())

    def train_model(self, epochs=2):
        self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.valid_dataset, verbose=2)

    def load_model(self):
        self.model = load_model(self.model_path)

    def save_model(self):
        save_model(self.model, self.model_path, overwrite=True)
    