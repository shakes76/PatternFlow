import tensorflow as tf
import numpy as np
from matplotlib import pyplot
from tensorflow.keras.losses import BinaryCrossentropy
import time
from tensorflow import random, train
from numpy.random import rand, randint
import os

from dataset import (
    load_images
)

from modules import (
    discriminator_model,
    generator_model,
    generator_optimizer,
    discriminator_optimizer,
)

# Define Constants
PIC_DIR = ["./keras_png_slices_data/keras_png_slices_data/keras_png_slices_test/", 
    "./keras_png_slices_data/keras_png_slices_data/keras_png_slices_train/", 
    "./keras_png_slices_data/keras_png_slices_data/keras_png_slices_validate/"]
EPOCHS = 120
LATENT_DIM = 256
NUM_BLOCKS = 7
INITIAL_SIZE = 4

def get_inputs(n, img_shape, latent_dim, n_style_block=7):
    """
    Create random inputs for the generator model in the form of [ones, z_space, noise]
    from gaussian normal distribution.
    """
    if rand() < 0.5:
        available_z = [random.normal((n, 1, latent_dim)) for _ in range(2)]
        z = tf.concat([available_z[randint(0, len(available_z))] for _ in range(n_style_block)], axis=1)
    else:
        z = tf.repeat(random.normal((n, 1, latent_dim)), n_style_block, axis=1)

    noise = random.normal([n, img_shape[0], img_shape[1], 1], 0, 1, tf.float32)
    return [tf.ones((n, 1)), z, noise]


def generate_real_samples(dataset, n_samples):
    """
    Select specified random samples of images from the given dataset. 
    Assign class label of 1 to the selected images. 
    """
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = tf.ones((n_samples, 1))
    return X, y


def create_checkpoint(gen_optimizer, disc_optimizer, generator, discriminator):
    """
    Create checkpoint for the generator and discriminator model. 
    """
    checkpoint = train.Checkpoint(generator_optimizer=gen_optimizer,
                                  discriminator_optimizer=disc_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)
    return checkpoint


def plot_model_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=9):
    """
    Generate and save 9 plotted fake images from the trained model.
    """
    # prepare fake examples
    generator_inputs = get_inputs(n_samples, dataset[0].shape, latent_dim)
    x_fake = g_model(generator_inputs, training=False)

    # save plot
    for i in range(3 * 3):
        pyplot.subplot(3, 3, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(x_fake[i], cmap="gray")
        
    # save plot to file
    filename = './images/generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()

def plot_model_loss(epoch, gen_losses, disc_real_losses, disc_fake_losses):
    """
    Plot model loss given the generator and discriminator's loss.
    """
    pyplot.plot(gen_losses)
    pyplot.plot(disc_real_losses)
    pyplot.plot(disc_fake_losses)

    pyplot.legend(["gen_loss", "disc_real_loss", "disc_fake_loss"])
    pyplot.show()

    # save plot to file
    filename = './images/Model_Loss.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()

@tf.function
def train_batch(real_samples, batch_size, latent_dim, g_model, d_model, generator_optimizer, discriminator_optimizer):
    """
    Train model using given batch_size, update the discriminator and generator's weights with learnt gradient. 
    """
    # Get random inputs for generator
    generator_inputs = get_inputs(batch_size, real_samples[0].shape,  latent_dim)

    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        fake_samples = g_model(generator_inputs, training = False)
        real_pred = d_model(real_samples, training = True)
        fake_pred = d_model(fake_samples, training = True)

        # Compare the real and fake predictions against their true labels. 
        generator_loss = BinaryCrossentropy()(tf.ones_like(fake_pred), fake_pred)
        real_loss = BinaryCrossentropy(label_smoothing=0.2)(tf.ones_like(real_pred), real_pred)
        fake_loss = BinaryCrossentropy(label_smoothing=0.2)(tf.zeros_like(fake_pred), fake_pred)
        discriminator_loss = real_loss + fake_loss 

    # Calculate gradient
    generator_grad = generator_tape.gradient(generator_loss, g_model.trainable_variables)
    discriminator_grad = discriminator_tape.gradient(discriminator_loss, d_model.trainable_variables)

    # Update discriminator and generator's weights with gradients.
    generator_optimizer.apply_gradients(zip(generator_grad, g_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_grad, d_model.trainable_variables))

    return generator_loss, real_loss, fake_loss


def train(data, epochs, latent_dim, g_model, d_model, gen_optimizer, disc_optimizer):
    """
    StyleGAN Training
    """
    # define the number of batch runs per epoch, and the batch size
    # depending on the batch_size the bat per epoch could be adjusted.
    bat_per_epo = 13000
    batch_size = 12 
    checkpoint = create_checkpoint(gen_optimizer, disc_optimizer, g_model, d_model)

    gen_losses, disc_real_losses, disc_fake_losses = [], [], []
    for epoch in range(epochs):
        start_time = time.time()
        for i in range(bat_per_epo):
            # Get randomly selected 'real' samples
            X_real, _ = generate_real_samples(data, batch_size)

            # Train model according to given half batch size
            gen_loss, disc_real_loss, disc_fake_loss = \
                train_batch(X_real, batch_size, latent_dim, g_model, d_model, gen_optimizer, disc_optimizer)

            # Get model summary every 100 batches.
            if i % 100 == 0:
                gen_loss = gen_loss.numpy()
                disc_real_loss= disc_real_loss.numpy()
                disc_fake_loss = disc_fake_loss.numpy()
                
                gen_losses.append(gen_loss)
                disc_real_losses.append(disc_real_loss)
                disc_fake_losses.append(disc_fake_loss)

                # Print training progress.
                plot_model_performance(epoch, g_model, d_model, X_real, latent_dim)

                print(
                    f'\rEpoch num: {epoch} / {epochs} | batch: {i} / {bat_per_epo} | \
                    gen_loss: {round(gen_loss, 4)} | disc_real_loss: {round(disc_real_loss, 4)} | \
                    disc_fake_loss: {round(disc_fake_loss, 4)} | \
                    Time taken: {round(((time.time() - start_time) / 60), 2)} min', end='')
                print()
        
        # Save checkpoint every 10 epochs.
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=os.path.join('./model/model_checkpoints', "ckpt"))
            np.save("g_losses", gen_losses)
            np.save("d_real_losses", disc_real_losses)
            np.save("d_fake_losses", disc_fake_losses)

            plot_model_loss(epoch, gen_losses, disc_real_losses, disc_fake_losses)


    return gen_losses, disc_real_losses, disc_fake_losses

def main():
    # Either load from the given directory or from the npy file
    images = load_images(PIC_DIR)
    # images = np.load("./oasis_data_grayscale.npy")

    g_model = generator_model(latent_dim = LATENT_DIM, num_blocks=NUM_BLOCKS, initial_size=INITIAL_SIZE)
    d_model = discriminator_model()

    # get generator optimizer, can pass in argument to specify the learning rate, beta_1 and beta_2
    gen_optimizer = generator_optimizer()
    disc_optimizer = discriminator_optimizer()
    gen_losses, disc_real_losses, disc_fake_losses= train(images, EPOCHS , LATENT_DIM, g_model, d_model, gen_optimizer, disc_optimizer)

    # save and plot model loss
    np.save("g_losses", gen_losses)
    np.save("d_real_losses", disc_real_losses)
    np.save('d_fake_losses',disc_fake_losses)
    plot_model_loss(EPOCHS, gen_losses, disc_real_losses, disc_real_losses)

if __name__ == '__main__':
    main()
        