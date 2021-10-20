import os.path

import tensorflow as tf
from time import time
from models import Generator, Discriminator
import matplotlib.pyplot as plt
import neptune.new as neptune
from datetime import datetime
from tensorflow.keras.utils import image_dataset_from_directory


def train_g(generator: Generator, discriminator: Discriminator, batch_size: int, latent_dim: int):
    latent = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as tape:
        fake = generator.model(latent, training=True)
        fake_score = discriminator.model(fake, training=False)
        loss = generator.loss(fake_score)

    gradient = tape.gradient(loss, generator.model.trainable_variables)
    generator.optimizer.apply_gradients(zip(gradient, generator.model.trainable_variables))

    score = tf.reduce_mean(fake_score)
    return score


def train_d(real, generator: Generator, discriminator: Discriminator, batch_size: int, latent_dim: int):
    latent = tf.random.normal([batch_size, latent_dim])
    with tf.GradientTape() as tape:
        fake = generator.model(latent, training=False)
        fake_score = discriminator.model(fake, training=True)
        real_score = discriminator.model(real, training=True)
        loss = discriminator.loss(real_score, fake_score)

    gradient = tape.gradient(loss, discriminator.model.trainable_variables)
    discriminator.optimizer.apply_gradients(zip(gradient, discriminator.model.trainable_variables))

    score = 1/2 * tf.reduce_mean(real_score) + 1/2 * tf.reduce_mean(1 - fake_score)
    return score


def show_images(generator: Generator, epoch, test_input, folder, save=True, rgb=False):
    predictions = generator.model(test_input, training=False)

    fig = plt.figure(figsize=(5, 5))

    if rgb:
        channels = 3
    else:
        channels = 1

    predictions = tf.reshape(predictions, (-1, 64, 64, channels))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)

        if rgb:
            plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5)
        else:
            plt.imshow(predictions[i, :, :, :] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    if save:
        path = os.path.join(folder, 'image_at_epoch_{}.png'.format(epoch))
        plt.savefig(path)

    plt.show()

    return fig


def train(dataset, generator: Generator, discriminator: Discriminator, validate_latent, log, output_folder: str,
          epochs: int, batch_size: int, latent_dim: int):
    iter = 0
    for epoch in range(epochs):
        start = time()

        for image_batch in dataset:
            # normalize to the range [-1, 1] to match the generator output
            image_batch = (image_batch - 255 / 2) / (255 / 2)

            d_score = train_d(image_batch, generator, discriminator, batch_size, latent_dim)
            g_score = train_g(generator, discriminator, batch_size, latent_dim)

            # log to neptune
            log["Generator_Score"].log(g_score)
            log["Discriminator_Score"].log(d_score)

            iter += 1

            # showing the result every 100 iterations
            if iter % 100 == 0:
                fig = show_images(generator, 0, validate_latent, output_folder, save=False)
                log["Validation"].upload(fig)

        # show and save the result every epoch
        show_images(generator, epoch, validate_latent, output_folder, save=True)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time() - start))
        print("Discriminator score: {}\t Generator score: {}".format(d_score, g_score))


if __name__ == "__main__":
    with open("neptune_credential.txt", 'r') as credential:
        token = credential.readline()

    run = neptune.init(
        project="zhien.zhang/styleGAN",
        api_token=token,
    )

    batch_size = 128
    alpha = 0
    lr = 0.0002
    beta_1 = 0.5
    num_of_epochs = 20
    latent_dim = 100
    num_examples_to_generate = 16
    log_freq = 5
    dropout = 0.3

    # setup models
    generator = Generator(alpha, lr, beta_1, latent_dim)
    generator.build()
    discriminator = Discriminator(dropout, lr, beta_1)
    discriminator.build()

    # prepare datasets
    image_folder = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_train"
    train_batches = image_dataset_from_directory(
        image_folder, labels=None, label_mode=None,
        class_names=None, color_mode='grayscale', batch_size=batch_size, image_size=(64, 64), shuffle=True, seed=None,
        validation_split=None, subset=None,
        interpolation='bilinear', follow_links=False,
        crop_to_aspect_ratio=False
    )

    # latent code for validation
    validation_latent = tf.random.normal([num_examples_to_generate, latent_dim], seed=1)

    # output folder
    run_folder = datetime.now().strftime("%d-%m/%Y_%H_%M_%S")
    upper_folder = "C:\\Users\\Zhien Zhang\\Desktop\\Other\\COMP3710\\StyleGAN\\Output\\image_in_training"
    output_folder = os.path.join(upper_folder, run_folder)
    os.makedirs(output_folder, exist_ok=True)

    train(train_batches, generator, discriminator, validation_latent, run, output_folder, num_of_epochs, batch_size,
          latent_dim)

    run.stop()
