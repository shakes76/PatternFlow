import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os

from glob import glob 
from time import time
from math import log2
from datetime import timedelta
from random import sample, random
from shutil import rmtree

from generator import Generator
from discriminator import Discriminator

EPOCHS = 20

IMAGE_SIZE = 64
CHANNELS = 1
LATENT_SIZE = 512

BLOCKS = int(log2(IMAGE_SIZE) - 1)
LEARNING_RATE = 0.0001
PENALTY = 10
MIXED_PROBABILITY = 0.9

BATCH_SIZE = 1
IMAGE_PATHS = glob('./keras_png_slices_data/keras_png_slices_test/*.png') \
    +  glob('./keras_png_slices_data/keras_png_slices_validate/*.png') \
    +  glob('./keras_png_slices_data/keras_png_slices_train/*.png')

CHECKPOINTS_PATH = './checkpoints'
SAMPLE_PATH = './samples'
CACHE_PATH = './cache/'
SAMPLE_SIZE = 1

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

        self.generator = Generator(self.image_size, BLOCKS, self.learning_rate, CHANNELS)

        self.discriminator = Discriminator(self.image_size, BLOCKS, self.learning_rate, CHANNELS)

        self.generator.build_model()
        self.discriminator.build_model()

        self.penalty_factor = PENALTY


    def load_data(self, image_paths): 
        image_tensor = tf.convert_to_tensor(image_paths, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(image_tensor)
        dataset = dataset.map(preprocessing, num_parallel_calls=8).cache(CACHE_PATH)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE).batch(BATCH_SIZE)
        self.dataset = dataset


    def train_step(self, batch, style, noise, penalty=True): 

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

            w = [self.generator.style(s) for s in style]

            fake_batch = self.generator.model(w + noise)

            real_predictions = self.discriminator.model(batch, training=True)
            fake_predictions = self.discriminator.model(fake_batch, training=True)

            generator_loss = self.generator.w_loss(fake_predictions)
            discriminator_loss = self.discriminator.w_loss(real_predictions, fake_predictions)

            if penalty:
                discriminator_loss += self.gradient_penalty(batch, fake_batch)
            
            generator_variables = self.generator.model.trainable_variables
            discriminator_variables = self.discriminator.model.trainable_variables

            self.generator.optimizer.apply_gradients(zip(generator_tape.gradient(generator_loss, generator_variables), generator_variables))
            self.discriminator.optimizer.apply_gradients(zip(discriminator_tape.gradient(discriminator_loss, discriminator_variables), discriminator_variables))

        return generator_loss, discriminator_loss


    def gradient_penalty(self, real_batch, fake_batch):

        alpha = tf.random.normal([BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
        interpolated = real_batch + alpha * (fake_batch - real_batch)

        with tf.GradientTape() as penalty_tape:

            penalty_tape.watch(interpolated)
            prediction = self.discriminator.model(interpolated, training=True)

        gradients = penalty_tape.gradient(prediction, [interpolated])[0]

        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        penalty = tf.reduce_mean((norm - 1.0) ** 2)

        return penalty * self.penalty_factor


    def train(self, epochs):

        train_start = time()

        fixed_style, fixed_noise = self.generate_style(0)

        for epoch in range(1, epochs + 1):

            style, noise = self.generate_style(MIXED_PROBABILITY)

            epoch_start = time()

            generator_losses = []
            discriminator_losses = []

            for batch in self.dataset:

                generator_loss, discriminator_loss = self.train_step(batch, style, noise)

                generator_losses.append(generator_loss)
                discriminator_losses.append(discriminator_loss)

            epoch_end = time()
            epoch_time = epoch_end - epoch_start

            print("Time taken for epoch" , str(epoch) , ":", str(timedelta(seconds=epoch_time)), \
                ", generator_loss =" , str(tf.get_static_value(tf.reduce_mean(generator_losses))), \
                ", discriminator_loss =" , str(tf.get_static_value(tf.reduce_mean(discriminator_losses))))

            generated_batch = self.generator.model.predict(fixed_style + fixed_noise)
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


    def generate_style(self, probability):

        style = [tf.random.normal([1, LATENT_SIZE], 0, 1)]
        noise = [tf.random.uniform([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1], 0, 1)]

        if random() < probability:
            seed = int(random() * BLOCKS)
            style = (seed * style) + [] + ((BLOCKS - seed) * style)
        else:
            style = BLOCKS * style

        return style, noise

        

def clear_samples(path):
    old_samples = glob(path + '/*')
    for sample in old_samples:
        rmtree(sample)
    print("Sample Cleared")


def clear_cache(path):
    rmtree(path)
    os.mkdir(path)
    print("Cache Cleared")


def main():

    args = parse_args()
    
    if args.clear == "cache":
        clear_cache(CACHE_PATH)
    elif args.clear == "samples":
        clear_samples(SAMPLE_PATH)
    elif args.clear == "both":
        clear_cache(CACHE_PATH)
        clear_samples(SAMPLE_PATH)

    gan = GAN(IMAGE_SIZE, LEARNING_RATE)
    print(gan.generator.model.summary())
    print(gan.discriminator.model.summary())

    gan.load_data(IMAGE_PATHS)

    if args.load == "weights":
        gan.load_weights(CHECKPOINTS_PATH)

    gan.train(EPOCHS)

        
def parse_args():

    parser = argparse.ArgumentParser(description='Train StyleGan2')

    parser.add_argument('--clear', required=False)
    parser.add_argument('--load', required=False)

    args =  parser.parse_args()

    return args 


if __name__ == '__main__':
    main()





