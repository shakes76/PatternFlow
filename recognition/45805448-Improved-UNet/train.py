import os
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import *
from modules import improved_unet

# Default values for ISIC 2016 Training Dataset
BATCH_SIZE = 5
IMAGE_SIZE = (192, 256)
EPOCHS = 20
TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1

PLOT_SAMPLES_PATH = 'pretraining_samples.png'

class Trainer:
    def __init__(self, images_path, masks_path, dataset_path, model_path, plot_samples_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.plot_samples_path = plot_samples_path

        self.batch_size = None
        self.image_height = None
        self.image_width = None

        self.full_dataset = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.model = None

    def load_data(self, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
        images = load_image_dataset_from_directory(self.images_path, image_size=image_size, batch_size=batch_size)
        masks = load_image_dataset_from_directory(self.masks_path, image_size=image_size, batch_size=batch_size, color_mode='grayscale')

        self.full_dataset = merge_image_mask_datasets(images, masks)
        self.full_dataset = preprocess_dataset(self.full_dataset)

    def load_existing_data(self):
        self.full_dataset = load_dataset(self.dataset_path)

    def save_data(self):
        save_dataset(self.full_dataset, self.dataset_path)

    def split_data(self, train_split=TRAIN_SPLIT, valid_split=VALID_SPLIT, test_split=TEST_SPLIT):
        num_samples = len(os.listdir(self.images_path))
        total = train_split + valid_split + test_split
        train_size = int(num_samples * (train_split / total) / BATCH_SIZE)
        valid_size = int(num_samples * (valid_split / total) / BATCH_SIZE)
        test_size = int(num_samples * (test_split / total) / BATCH_SIZE)

        self.train_dataset = self.full_dataset.take(train_size)
        self.valid_dataset = self.full_dataset.skip(train_size).take(valid_size)
        self.test_dataset = self.full_dataset.skip(train_size + valid_size).take(test_size)

        for batch in self.test_dataset.take(1):
            test_images, test_masks = batch[0], batch[1]
            self.batch_size, self.image_height, self.image_width, _ = list(tf.shape(test_images))
            print(f'Shape of input: {tf.shape(test_images)}')
            print(f'Shape of output: {tf.shape(test_masks)}')

        print(f'Training-Validation-Testing Batch Split: ' +
                f'{self.train_dataset.cardinality().numpy()}-' +
                f'{self.valid_dataset.cardinality().numpy()}-' +
                f'{self.test_dataset.cardinality().numpy()}\n' +
                f'Batch size: {self.batch_size}')

    def output_samples(self):
        for batch in self.test_dataset.take(1):
            test_images, test_masks = batch[0], tf.argmax(batch[1], axis=-1)

            plt.figure(figsize=(10,10))
            for i in range(self.batch_size):
                plt.subplot(self.batch_size, 2, i*2 + 1)
                plt.imshow(test_images[i])
                plt.axis('off')
                plt.title('Image')

                plt.subplot(self.batch_size, 2, i*2 + 2)
                plt.imshow(test_masks[i], vmin=0, vmax=1)
                plt.axis('off')
                plt.title('Actual Mask')

            plt.savefig(self.plot_samples_path)

    def build_model(self):
        self.model = improved_unet((self.image_height, self.image_width, 3), self.batch_size)

    def summarise_model(self):
        print(self.model.summary())

    def train_model(self, epochs=EPOCHS):
        self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.valid_dataset, verbose=1)

    def load_model(self):
        if self.model == None:
            self.build_model()
        self.model.load_weights(self.model_path)

    def save_model(self):
        self.model.save_weights(self.model_path, overwrite=True)


def train_isic_dataset(images_path='', masks_path='', dataset_path='', model_path='', plot_samples_path=PLOT_SAMPLES_PATH,
                        override_dataset=False, override_samples=False, override_model=False):
    
    trainer = Trainer(images_path, masks_path, dataset_path, model_path, plot_samples_path)

    if override_dataset or not os.path.isdir(trainer.dataset_path):
        trainer.load_data()
        trainer.save_data()
    else:
        trainer.load_existing_data()

    trainer.split_data()

    if override_samples or not os.path.exists(trainer.plot_samples_path):
        trainer.output_samples()

    if override_model or not os.path.isdir(trainer.model_path):
        trainer.build_model()
        trainer.summarise_model()
        trainer.train_model()
        trainer.save_model()
    else:
        trainer.load_model()
        trainer.summarise_model()

    return trainer
        