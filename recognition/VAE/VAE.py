import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, InputLayer, Flatten, Dense, Reshape, BatchNormalization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import os
from tensorflow.keras.models import load_model
import math


class VAE(tf.keras.Model):
    def __init__(self, latent_dimsion):
        super(VAE, self).__init__()
        self.latent_dim = latent_dimsion
        self.encoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(256, 256, 1)),
                Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu'),
                BatchNormalization(),
                Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                BatchNormalization(),
                Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                BatchNormalization(),
                Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
                BatchNormalization(),
                Flatten(),
                # No activation
                Dense(latent_dimsion * 2),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dimsion,)),
                Dense(units=32 * 32 * 32, activation=tf.nn.relu),
                BatchNormalization(),
                Reshape(target_shape=(32, 32, 32)),
                Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu'),
                BatchNormalization(),
                Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                BatchNormalization(),
                Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                BatchNormalization(),
                Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu'),
                BatchNormalization(),
                # No activation
                Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, sigmoid=True)

    def encode(self, x):
        parameters = self.encoder(x)
        mean, log_var = tf.split(parameters, num_or_size_splits=2, axis=1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        # generate epsilon from a standard normal distribution
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(log_var * .5) + mean

    def decode(self, z, sigmoid=False):
        logits = self.decoder(z)
        if sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def generate_images(self, test_sample):
        mean, log_var = model.encode(tf.expand_dims(test_sample, axis=-1))
        z = model.reparameterize(mean, log_var)
        predictions = model.sample(z)
        return predictions


def get_test_sample(test_dataset):
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:batch_size, :, :, :]
        return test_sample


def load_img_to_tensor(ds, crop_ratio):
    list_of_batches = list(ds.as_numpy_iterator())
    brains = list()
    for batch in list_of_batches:
        for images in batch:
            for i in range(images.shape[0]):
                if (len(images[i].shape) == 3):
                    image = tf.image.central_crop(images[i], crop_ratio)/255
                    brains.append(image)
    brain_images = tf.convert_to_tensor(brains, dtype=tf.float32)
    return brain_images


def get_dataset(train_dir, test_dir, batch_size, crop_ratio=1):
    # load the images to BatchDataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir, color_mode='grayscale')
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(test_dir, color_mode='grayscale')
    train_dataset = load_img_to_tensor(train_dataset, crop_ratio)
    test_dataset = load_img_to_tensor(test_dataset, crop_ratio)
    train_size = train_dataset.shape[0]
    train_dataset = (tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(train_size).batch(batch_size))
    return train_dataset, test_dataset