#imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

import modules.layers as layers
import modules.losses as losses

# Check Tensorflow version
print("Tensorflow version " + tf.__version__)

#parameters
PATH = '/content/drive/My Drive/Datasets/keras_png_slices_data/keras_png_slices_train/*'
BATCH_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = [256, 256]
EPOCHS = 1200

# Load the dataset
print('Loading the OASIS Brain dataset.....')
image_file_paths = tf.io.gfile.glob(PATH)
BUFFER_SIZE = len(image_file_paths) + 1
#shuffle the image file paths.
image_file_paths = tf.data.Dataset.from_tensor_slices(image_file_paths).shuffle(BUFFER_SIZE)

# Preprocess data
def parse_image(filename):

  image = tf.io.read_file(filename)
  image = tf.image.decode_png(image, channels = 0) #keep original color channels.
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = (image - 127.5) / 127.5

  return tf.image.resize(image, IMG_SIZE)

#map over each filepath to training images and parse, cache and batch the images.
training_ds = image_file_paths.map(parse_image, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

#check batch size and image shape.
image_batch = next(iter(training_ds))
no_inputs, input_width, input_height, input_channels = image_batch.shape

#model parameters
noise_dimensions = 100
gen_input_shape = (noise_dimensions, )
disc_input_shape = (input_width, input_height, input_channels)

# Build the networks
def generator_network(input_shape):
    '''
    Receive random noise with gaussian distribution.
    Outputs image.
    '''
    input = Input(shape=input_shape)

    dense = layers.FullyConnected(input, 8*8*512, reshape_shape=(8,8,512))
    
    conv1 = layers.Generator_Norm_Conv2DTranspose(dense, filters=512)
    conv2 = layers.Generator_Norm_Conv2DTranspose(conv1, filters=256)
    conv3 = layers.Generator_Norm_Conv2DTranspose(conv2, filters=128)
    conv4 = layers.Generator_Norm_Conv2DTranspose(conv3, filters=128)
    conv5 = layers.Generator_Tanh_Conv2DTranspose(conv4, filters=1)
   
    return Model(inputs=input, outputs=conv5, name="generator")

def discriminator_network(input_shape):
    '''
    Receive generator output image and real images from dataset.
    Outputs binary classficiation.
    '''
    input = Input(shape=input_shape)
    conv1 = layers.Discriminator_Norm_Dropout_Conv2D(input, filters=64, dropout=0.15)
    conv2 = layers.Discriminator_Norm_Dropout_Conv2D(conv1, filters=128, dropout=0.15)
    conv3 = layers.Discriminator_Norm_Dropout_Conv2D(conv2, filters=256, dropout=0.15)
    conv4 = layers.Discriminator_Norm_Dropout_Conv2D(conv3, filters=512, dropout=0.1)
    output = layers.Flatten_Dense(conv4)

    return Model(inputs=input, outputs=output, name="discriminator")


#build generator
generator = generator_network(gen_input_shape)
generator.summary()

#build discriminator
discriminator = discriminator_network(disc_input_shape)
discriminator.summary()

#generator loss
def gen_loss(generated_outputs):
    return losses.generator_crossentropy(generated_outputs)

#discriminator loss
def disc_loss(generated_outputs, real_outputs):
    return losses.discriminator_crossentropy(generated_outputs, real_outputs)

#discriminator fake detection accuracy with threshold > 0.5
def disc_acc(generated_outputs, real_outputs):
    return losses.discriminator_accuracy(generated_outputs, real_outputs)

#optimizers
gen_opt = Adam(learning_rate = 0.0002)
disc_opt = Adam(learning_rate = 0.0001)

#prediction parameters
no_gen_images = 9
seed = tf.random.normal([no_gen_images, noise_dimensions]) #allows us to track progress of generated brains.
generated_images_path = '/content/drive/My Drive/dilated_test/'

#Training
@tf.function # compile the function for faster training.
def train_step_dcgan(image_batch):
    noise = tf.random.normal([BATCH_SIZE, noise_dimensions])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_outputs = discriminator(image_batch, training=True)
        generated_outputs = discriminator(generated_images, training=True)

        generator_loss = gen_loss(generated_outputs)
        discriminator_loss = disc_loss(generated_outputs, real_outputs)
        generated_acc, real_acc = disc_acc(generated_outputs, real_outputs)
    
    gradients_of_gen = gen_tape.gradient(generator_loss, generator.trainable_variables)
    gradients_of_disc = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_of_disc, discriminator.trainable_variables))

    return generator_loss, discriminator_loss, generated_acc, real_acc

def train(dataset, epochs):
    gen_losses = []
    disc_losses = []
    generated_accuracy = []
    real_accuracy = []
    
    for epoch in range(1, epochs+1):
        start = time.time()

        for image_batch in dataset:
            generator_loss, discriminator_loss, generated_acc, real_acc = train_step_dcgan(image_batch)

        
        
        print('Time for epoch {} (disc_loss {}, gen_loss {}) is {} sec'.format(epoch, 
                                                                               discriminator_loss, 
                                                                               generator_loss, 
                                                                               time.time()-start))
        print('Disc generated acc {}, real acc {}'.format(generated_acc, real_acc))
        # Produce images for the GIF as we go
        generate_and_save_images(generator, epoch, seed)                                         
        gen_losses.append(generator_loss)
        disc_losses.append(discriminator_loss)
        generated_accuracy.append(generated_acc)
        real_accuracy.append(real_acc)

    return gen_losses, disc_losses, generated_accuracy, real_accuracy
        

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    
    fig = plt.figure(figsize=(12, 12))

    for i in range(predictions.shape[0]):
        plt.subplot(3, 3, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
        
    plt.savefig(generated_images_path + 'epoch_{:04d}.png'.format(epoch), transparent=True)
    plt.show()

#training
all_losses = train(training_ds, EPOCHS)

#plot
plt.plot(np.arange(EPOCHS), all_losses[0], label = 'gen_loss')
plt.plot(np.arange(EPOCHS), all_losses[1], label = 'disc_loss')
plt.xlabel('epochs (64 Images per epoch)')
plt.ylabel('Binary Cross Entropy Loss')
plt.title('DCGAN Cross Entropy Loss')
plt.legend()
plt.savefig(generated_images_path+'Cross_Entropy.png')

print('END')