import tensorflow as tf
import numpy as np
from matplotlib import pyplot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import time
from IPython import display
from tensorflow import random, train
from numpy.random import randn, rand, randint
import os

from dataset import (
    load_images
)

from modules import (
    discriminator_model,
    generator_model,
)

def get_inputs(n, img_shape, latent_dim, n_style_block=7):
    if rand() < 0.5:
        available_z = [random.normal((n, 1, latent_dim)) for _ in range(2)]
        z = tf.concat([available_z[randint(0, len(available_z))] for _ in range(n_style_block)], axis=1)
    else:
        z = tf.repeat(random.normal((n, 1, latent_dim)), n_style_block, axis=1)

    noise = random.normal([n, img_shape[0], img_shape[1], 1], 0, 1, tf.float32)
    return [tf.ones((n, 1)), z, noise]


def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = tf.ones((n_samples, 1))
    return X, y


def create_checkpoint(gen_optimizer, disc_optimizer, generator, discriminator):
    checkpoint = train.Checkpoint(generator_optimizer=gen_optimizer,
                                  discriminator_optimizer=disc_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)
    return checkpoint


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=9):
    # prepare fake examples
    generator_inputs = get_inputs(n_samples, dataset[0].shape, latent_dim)
    x_fake = g_model(generator_inputs, training=False)

    # save plot
    for i in range(3 * 3):
        # define subplot
        pyplot.subplot(3, 3, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(x_fake[i], cmap="gray")
    # save plot to file
    filename = './images/generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()

@tf.function
def train_batch(real_samples, batch_size, latent_dim, g_model, d_model, generator_optimizer, discriminator_optimizer):
    generator_inputs = get_inputs(batch_size, real_samples[0].shape,  latent_dim)

    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        fake_samples = g_model(generator_inputs, training = True)
        real_pred = d_model(real_samples, training = True)
        fake_pred = d_model(fake_samples, training = True)

        # Compare the real and fake predictions against their true labels. 
        generator_loss = BinaryCrossentropy()(tf.ones_like(fake_pred), fake_pred)
        real_loss = BinaryCrossentropy(label_smoothing=0.2)(tf.ones_like(real_pred), real_pred)
        fake_loss = BinaryCrossentropy(label_smoothing=0.2)(tf.zeros_like(fake_pred), fake_pred)
        discriminator_loss = real_loss + fake_loss 

    generator_grad = generator_tape.gradient(generator_loss, g_model.trainable_variables)
    discriminator_grad = discriminator_tape.gradient(discriminator_loss, d_model.trainable_variables)

    # Update discriminator and generator's weights with gradients.
    generator_optimizer.apply_gradients(zip(generator_grad, g_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_grad, d_model.trainable_variables))

    return generator_loss.numpy(), real_loss.numpy(), fake_loss.numpy()


def train(data, epochs, latent_dim, g_model, d_model, gen_optimizer, disc_optimizer):
    bat_per_epo = 10000 #int(data.shape[0] / epochs)
    half_batch = 12 #int(epochs / 2)
    checkpoint = create_checkpoint(gen_optimizer, disc_optimizer, g_model, d_model)

    g_losses, d_real_losses, d_fake_losses = [], [], []
    for epoch in range(epochs):
        start_time = time.time()
        for i in range(bat_per_epo):
            # Get randomly selected 'real' samples
            X_real, _ = generate_real_samples(data, half_batch)

            # Train model according to given half batch size
            g_loss, d_real_loss, d_fake_loss = train_batch(X_real, half_batch, latent_dim, g_model, d_model, gen_optimizer, disc_optimizer)

            # Get model summary every 100 batches.
            if i % 100 == 0:
                g_losses.append(g_loss)
                d_real_losses.append(d_real_loss)
                d_fake_losses.append(d_fake_loss)

                # Print training progress.
                summarize_performance(epoch, g_model, d_model, X_real, latent_dim)

                print(
                    f'\rEpoch: {epoch} / {epochs} | batch: {i} / {bat_per_epo} | '
                    f'g_loss: {round(g_loss, 4)} | d_fake_loss: {round(d_fake_loss, 4)} | '
                    f'd_real_loss: {round(d_real_loss, 4)} | '
                    f'Time taken: {round(((time.time() - start_time) / 60), 2)} minutes', end='')
                print()
        
        # Save checkpoint every 10 epochs.
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=os.path.join('./model/model_checkpoints', "ckpt"))

    return g_losses, d_real_losses, d_fake_losses

def main():
    PIC_DIR = ["./keras_png_slices_data/keras_png_slices_data/keras_png_slices_test/", 
            "./keras_png_slices_data/keras_png_slices_data/keras_png_slices_train/", 
            "./keras_png_slices_data/keras_png_slices_data/keras_png_slices_validate/"]
    # images = load_images(PIC_DIR)

    # Either load from the given directory or from the npy file
    images = np.load("./oasis_data_grayscale.npy")
    epochs = 100
    latent_dim = 256

    g_model = generator_model(latent_dim = latent_dim)
    d_model = discriminator_model()
    gen_optimizer = Adam(learning_rate=2e-7, beta_1=0.5, beta_2=0.99)
    disc_optimizer = Adam(learning_rate=1.5e-7, beta_1=0.5, beta_2=0.99)
    g_losses, d_real_losses, d_fake_losses= train(images, epochs , latent_dim, g_model, d_model, gen_optimizer, disc_optimizer)

    np.save("g_losses", g_losses)
    np.save("d_real_losses", d_real_losses)
    np.save('d_fake_losses',d_fake_losses)

if __name__ == '__main__':
    main()
        