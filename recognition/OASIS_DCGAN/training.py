import os
import time
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt

import main
import dcgan as dcgan
from config import Config

# Set model
generator = dcgan.make_generator_model()
discriminator = dcgan.make_discriminator_model()
# generator.summary()
# discriminator.summary()
# Set optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Set checkpoint
checkpoint_prefix = os.path.join(Config.CHECKPOINT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

SEED = tf.random.normal([16, Config.NOISE_DIM])  # generate 16 examples


@tf.function
def train_step(images):
    noise = tf.random.normal([Config.BATCH_SIZE, Config.NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = dcgan.generator_loss(fake_output)
        disc_loss = dcgan.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    plt.clf()  # clean fig
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(((predictions[i] + 1.0) / 2.0 * 255), cmap='gray')
        plt.axis('off')
        plt.savefig('gen_im\image_at_epoch_{:03d}.jpg'.format(epoch))
        plt.pause(0.25)
    plt.pause(2)


# restore latest checkpoint
def restore_checkpoint():
    checkpoint.restore(tf.train.latest_checkpoint(Config.CHECKPOINT_DIR))


def train(dataset, epochs):
    plt.figure(figsize=(4, 4))
    plt.ion()
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # display figure
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, SEED)

        # save model for each 10 epoch
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    plt.ioff()
