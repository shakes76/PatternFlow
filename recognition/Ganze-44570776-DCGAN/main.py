"""
This is DCGAN tensorflow implementation for COMP3710 Report, resolving problem 6.
[Create a generative model of the OASIS brain or the OAI AKOA knee data set using a DCGAN that
has a “reasonably clear image” and a Structured Similarity (SSIM) of over 0.6.]

@author: Ganze Zheng
@studentID: 44570776
@date: 04/11/2020
"""

import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from PIL import Image

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# parameters
epochs = 100
batch_size = 64
dataset_path = "./keras_png_slices_data/keras_png_slices_train/*"
output_path = "./keras_png_slices_data/"

# Load OASIS Dataset
filenames = glob.glob(dataset_path)

if len(filenames) == 0:
    print("Error! Images not loaded!")
    exit()
else:
    n_datasize = len(filenames)

images = list()

for i in range(n_datasize):
    images.append(np.asarray(Image.open(filenames[i])))

images = np.array(images)

# Check dataset shape and verify the images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i], cmap='gray')
plt.savefig(output_path + 'Original_Images.png')

# Normalise data to [0, 1]
data = images.reshape((n_datasize, 256, 256, 1)).astype('float32')
data = data / 255.0

# Batch and shuffle the data
X_train = tf.data.Dataset.from_tensor_slices(
    data).shuffle(n_datasize).batch(batch_size)


# Build networks
# Generator network
def generator_network(input_shape, name='G'):
    """
    Generator network
    """
    input = Input(input_shape)
    dense = Dense(16*16*256)(input)
    norm = BatchNormalization()(dense)
    activation = LeakyReLU()(norm)
    dense = Reshape((16, 16, 256))(activation)
    # 16x16x256
    deepConvNet = Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same')(dense)
    norm = BatchNormalization()(deepConvNet)
    activation = LeakyReLU()(norm)
    # 32x32x128
    deepConvNet = Conv2DTranspose(
        64, (4, 4), strides=(2, 2), padding='same')(activation)
    norm = BatchNormalization()(deepConvNet)
    activation = LeakyReLU()(norm)
    # 64x64x64
    deepConvNet = Conv2DTranspose(
        32, (4, 4), strides=(2, 2), padding='same')(activation)
    norm = BatchNormalization()(deepConvNet)
    activation = LeakyReLU()(norm)
    # 128x128x32
    g_net = Conv2DTranspose(1, (4, 4), strides=(
        2, 2), padding='same', activation='sigmoid')(activation)
    # 256x256x1

    return Model(inputs=input, outputs=g_net, name=name)


# Discriminator network
def discriminator_network(input_shape, name='D'):
    """
    Discriminator network
    """
    input = Input(input_shape)
    # 256x256x1
    convNet = Conv2D(32, (4, 4), strides=(2, 2), padding='same')(input)
    activation = LeakyReLU()(convNet)
    norm = BatchNormalization()(activation)
    # 128x128x32
    convNet = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(norm)
    activation = LeakyReLU()(convNet)
    norm = BatchNormalization()(activation)
    # 64x64x64
    convNet = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(norm)
    activation = LeakyReLU()(convNet)
    norm = BatchNormalization()(activation)
    # 32x32x128

    flatten = Flatten()(norm)
    d_net = Dense(1, activation='sigmoid')(flatten)

    return Model(inputs=input, outputs=d_net, name=name)


# Build models
noise_shape = (100, )
image_shape = (256, 256, 1)

# Build generator
generator = generator_network(noise_shape)
generator.summary()

# Build discriminator
discriminator = discriminator_network(image_shape)
discriminator.summary()

# Model losses and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


# Discriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# Generator loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.SGD()
discriminator_optimizer = tf.keras.optimizers.SGD()

# Training
num_examples_to_generate = 5
seed = tf.random.normal([num_examples_to_generate, 100])


# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


# Main training loop
def train(dataset, epochs):
    g_losses = []
    d_losses = []

    for epoch in range(epochs):
        start = time.time()

        g_loss = -1
        d_loss = -1
        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch)

        print('Time for epoch {} (g_loss {}, d_loss {}) is {} sec'.format(
            epoch + 1, g_loss, d_loss, time.time()-start))
        g_losses.append(g_loss)
        d_losses.append(d_loss)

        # Check the output of the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            generate_sample_images(generator, epoch, seed)

    return g_losses, d_losses


def generate_sample_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(15, 15))
    for i in range(predictions.shape[0]):
        plt.subplot(1, 5, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
    
    plt.savefig(output_path + 'image_at_epoch_{:04d}.png'.format(epoch + 1))


# Train the model and show the results
G_losses, D_losses = train(X_train, epochs)

# Loss Curve
plt.figure(figsize=(10, 10))
plt.plot(G_losses, color='red', label='Generator_loss')
plt.plot(D_losses, color='blue', label='Discriminator_loss')
plt.legend()
plt.xlabel('total batches')
plt.ylabel('loss')
plt.savefig(output_path+'Loss_Curve.png')

# Predict
n_images = 25
noise = tf.random.normal([n_images, 100])
im = generator(noise, training=False)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(im[i, :, :, 0], cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.savefig(output_path + 'Examples.png')

print("End")
