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

class Trainer:
    def __init__(self, images_path, masks_path, dataset_path, model_path, plots_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.plots_path = plots_path

        self.batch_size = None
        self.image_height = None
        self.image_width = None

        self.full_dataset = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.model = None
        self.history = None

    def load_data(self, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
        """
        Loads images at a previously given path into a dataset, then preprocesses the dataset.
        """
        images = load_image_dataset_from_directory(self.images_path, image_size=image_size, batch_size=batch_size)
        masks = load_image_dataset_from_directory(self.masks_path, image_size=image_size, batch_size=batch_size, color_mode='grayscale')

        self.full_dataset = merge_image_mask_datasets(images, masks)
        self.full_dataset = preprocess_dataset(self.full_dataset)

    def load_existing_data(self):
        """
        Loads an existing preprocessed dataset into memory.
        """
        self.full_dataset = load_dataset(self.dataset_path)

    def save_data(self):
        """
        Saves the preprocessed dataset at a previously given path.
        """
        save_dataset(self.full_dataset, self.dataset_path)

    def split_data(self, train_split=TRAIN_SPLIT, valid_split=VALID_SPLIT, test_split=TEST_SPLIT):
        """
        Splits the full dataset into three partitions for training, validation, and testing.
        """
        num_samples = len(os.listdir(self.images_path))
        total = train_split + valid_split + test_split
        train_size = int(num_samples * (train_split / total) / BATCH_SIZE)
        valid_size = int(num_samples * (valid_split / total) / BATCH_SIZE)
        test_size = int(num_samples * (test_split / total) / BATCH_SIZE)

        self.train_dataset = self.full_dataset.take(train_size)
        self.valid_dataset = self.full_dataset.skip(train_size).take(valid_size)
        self.test_dataset = self.full_dataset.skip(train_size + valid_size).take(test_size)

    def summarise_data(self):
        """
        Prints information about the data such as the training-validation-testing split, and the input
        and output shapes of the model.
        """
        for batch in self.test_dataset.take(1):
            test_images, test_masks = batch[0], batch[1]
            self.batch_size, self.image_height, self.image_width, _ = test_images.get_shape()
            print(f'Training-Validation-Testing batch split: ' +
                    f'{self.train_dataset.cardinality().numpy()}-' +
                    f'{self.valid_dataset.cardinality().numpy()}-' +
                    f'{self.test_dataset.cardinality().numpy()}')
            print(f'Input shape: {tf.shape(test_images)}')
            print(f'Output shape: {tf.shape(test_masks)}')

    def plot_data(self):
        """
        Plots images from one batch of the testing dataset and saves it to storage. This function is
        useful for ensuring that the dataset being fed into the model is correct.
        """
        if not os.path.isdir(self.plots_path):
            os.makedirs(self.plots_path)

        for batch in self.test_dataset.take(1):
            test_images, test_masks = batch[0], tf.argmax(batch[1], axis=-1)

            plt.figure(figsize=(8,8))
            for i in range(self.batch_size):
                plt.subplot(self.batch_size, 2, i*2 + 1)
                plt.imshow(test_images[i])
                plt.axis('off')
                plt.title('Image')

                plt.subplot(self.batch_size, 2, i*2 + 2)
                plt.imshow(test_masks[i], vmin=0, vmax=1)
                plt.axis('off')
                plt.title('Actual Mask')

            plt.savefig(self.plots_path + '/preprocessed_samples.png')
            print(f'Saved plot to {self.plots_path}/preprocessed_samples.png')

    def build_model(self):
        """
        Builds the Improved UNet model from existing modules.
        """
        self.model = improved_unet((self.image_height, self.image_width, 3), self.batch_size)

    def summarise_model(self):
        """
        Prints a summary of the built model to stdout.
        """
        print(self.model.summary())

    def train_model(self, epochs=EPOCHS):
        """
        Trains the model on the training dataset and validating on the validation set. Saves the History callback
        for later.
        """
        self.history = self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.valid_dataset, verbose=2)

    def plot_model(self):
        """
        Plots the dice coefficient over time from the History callback saved from fitting the model.
        """
        if not os.path.isdir(self.plots_path):
            os.makedirs(self.plots_path)

        plt.figure(figsize=(8,8))
        plt.title(f'Model Dice Coefficient')
        plt.plot(self.history.history['dice_coefficient'])
        plt.plot(self.history.history[f'val_dice_coefficient'])
        plt.xlabel('Epoch')
        plt.ylabel('Dice Coefficient')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.savefig(self.plots_path + f'/model_dice_coefficient.png')
        print(f'Saved plot to {self.plots_path}/model_dice_coefficient.png')

    def load_model(self):
        """
        Builds a model if one doesn't exist, then loads the existing, trained model weights into memory.
        """
        if self.model == None:
            self.build_model()

        self.model.load_weights(self.model_path + '/model')

    def save_model(self):
        """
        Saves a trained model's weights into storage.
        """
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        self.model.save_weights(self.model_path + '/model', overwrite=True)


def train_isic_dataset(images_path='', masks_path='', dataset_path='', model_path='', plots_path='',
                        override_dataset=False, override_samples=False, override_model=False):
    """
    Main driver function for training the Improved UNet model on the ISIC dataset.
    """
    
    trainer = Trainer(images_path, masks_path, dataset_path, model_path, plots_path)

    if override_dataset or not os.path.isdir(trainer.dataset_path):
        trainer.load_data()
        trainer.save_data()
    else:
        trainer.load_existing_data()

    trainer.split_data()
    trainer.summarise_data()

    if override_samples or not os.path.isdir(trainer.plots_path):
        trainer.plot_data()

    if override_model or not os.path.isdir(trainer.model_path):
        trainer.build_model()
        trainer.summarise_model()
        trainer.train_model()
        trainer.save_model()
        trainer.plot_model()
    else:
        trainer.load_model()

    return trainer
        