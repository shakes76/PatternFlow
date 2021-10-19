import matplotlib.pyplot as plt
import tensorflow as tf
import os

from glob import glob
from time import time
from datetime import timedelta
from random import sample
from shutil import rmtree

from generator import Generator
from discriminator import Discriminator


EPOCHS = 5

IMAGE_SIZE = 64
CHANNELS = 1
LATENT_SIZE = 100

BLOCKS = 4
LEARNING_RATE = 0.00015
BETA = 0.5

BATCH_SIZE = 32
IMAGE_PATHS = glob('./keras_png_slices_data/keras_png_slices_test/*.png') \
    +  glob('./keras_png_slices_data/keras_png_slices_validate/*.png') \
    +  glob('./keras_png_slices_data/keras_png_slices_train/*.png')

CHECKPOINTS_PATH = './checkpoints'
SAMPLE_PATH = './samples'
SAMPLE_SIZE = 10

def preprocessing(path):
    image = tf.image.decode_png(tf.io.read_file(path), channels=1)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image


class GAN():


    def __init__(self, image_size, learning_rate):
        self.image_size = image_size
        self.learning_rate = learning_rate

        self.dataset = None
        self.generator = Generator(self.image_size, self.learning_rate, BETA, BLOCKS, CHANNELS)
        self.discriminator = Discriminator(self.image_size, self.learning_rate, BETA)

        self.generator.build_model()
        self.discriminator.build_model()


    def load_data(self, image_paths):
        image_tensor = tf.convert_to_tensor(image_paths, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(image_tensor)
        dataset = dataset.map(preprocessing, num_parallel_calls=8).cache('./cache/')
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE).batch(BATCH_SIZE)
        self.dataset = dataset


    def train_step(self, batch): 
        seed = tf.random.normal([BATCH_SIZE, LATENT_SIZE])

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

            fake_batch = self.generator.model(seed, training=True)

            real_predictions = self.discriminator.model(batch, training=True)
            fake_predictions = self.discriminator.model(fake_batch, training=True)

            generator_loss = self.generator.loss(fake_predictions)
            discriminator_loss = self.discriminator.loss(real_predictions, fake_predictions)
            
            generator_variables = self.generator.model.trainable_variables
            discriminator_variables = self.discriminator.model.trainable_variables

            self.generator.optimizer.apply_gradients(zip(generator_tape.gradient(generator_loss, generator_variables), generator_variables))
            self.discriminator.optimizer.apply_gradients(zip(discriminator_tape.gradient(discriminator_loss, discriminator_variables), discriminator_variables))

        return generator_loss, discriminator_loss


    def train(self, epochs):

        train_start = time()

        fixed_seed = tf.random.normal([self.image_size * self.image_size, LATENT_SIZE], 0, 1)
        
        for epoch in range(1, epochs + 1):

            epoch_start = time()

            generator_losses = []
            discriminator_losses = []

            for batch in self.dataset:

                generator_loss, discriminator_loss = self.train_step(batch)

                generator_losses.append(generator_loss)
                discriminator_losses.append(discriminator_loss)

            epoch_end = time()
            epoch_time = epoch_end - epoch_start

            print("Time taken for epoch" , str(epoch) , ":", str(timedelta(seconds=epoch_time)), \
                ", generator_loss =" , str(tf.get_static_value(tf.reduce_mean(generator_losses))), \
                ", discriminator_loss =" , str(tf.get_static_value(tf.reduce_mean(discriminator_losses))))

            generated_batch = self.generator.model.predict(fixed_seed)
            self.save_samples(SAMPLE_PATH, generated_batch, epoch)

            self.save_weights(CHECKPOINTS_PATH, epoch)

        train_end = time()
        train_time = train_end - train_start

        print("Training Time :", str(timedelta(seconds=train_time)))


    def save_weights(self, path, epoch):
        self.generator.model.save_weights(path + '/generator/epoch' + str(epoch) + '.ckpt')
        self.discriminator.model.save_weights(path + '/discriminator/epoch' + str(epoch) + '.ckpt')


    def load_weights(self, path):
        self.generator.model.load_weights(tf.train.latest_checkpoint(path + '/generator'))
        self.discriminator.model.load_weights(tf.train.latest_checkpoint(path + '/discriminator'))


    def save_samples(self, path, generated_batch, epoch):

        random_index = sample(range(len(generated_batch)), SAMPLE_SIZE)

        if not os.path.exists(path + '/epoch' + str(epoch)):
            os.mkdir(path + '/epoch' + str(epoch))

        for index in random_index:
            plt.imshow(generated_batch[index][:,:,0])
            plt.axis('off')
            plt.savefig(path + '/epoch' + str(epoch) + '/' + str(index) + '.png')

        

def clear_samples(path):
    old_samples = glob(path + '/*')
    for sample in old_samples:
        rmtree(sample)





clear_samples(SAMPLE_PATH)
gan = GAN(IMAGE_SIZE, LEARNING_RATE)
gan.load_data(IMAGE_PATHS)
gan.load_weights(CHECKPOINTS_PATH)
gan.train(EPOCHS)







"""
for batch in dataset:
    break
"""

