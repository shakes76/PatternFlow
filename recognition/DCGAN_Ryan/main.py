import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import glob
import imageio
from IPython import display
from statistics import mean
from model import *


def load_and_preprocess_image(name):
    """
    load the images based on the file path and resize the size into 256*256
    """
    image = tf.io.read_file('./keras_png_slices_data/keras_png_slices_train/' + name)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize_with_pad(image, 256, 256)
    image = (image - 127.5) / 127.5
    return image
# Define the training loop
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def generate_and_save_images(model, epoch, test_input):
    """
    Generate and save images
    """
    predictions = model(test_input, training=False)
    print(predictions.shape)
    fig = plt.figure(figsize=(10,10))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def train(dataset, epochs):
      for epoch in range(epochs):
            start = time.time()
            
            for image_batch in dataset:
                train_step(image_batch)

            # going to generate GIF
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                     epoch + 1,
                                     seed)

            # save once every 15 epoches
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

            # generate and save the image at the end of the last epoch
            display.clear_output(wait=True)
            generate_and_save_images(generator,epochs,seed)
            
def max_ssim(generated_image):
    """
    measure the generated image by calculating the max value of SSIM
    """
    SSIMs=[]
    for name in names:
        image = tf.io.read_file('./keras_png_slices_data/keras_png_slices_train/' +name)
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize_with_pad(image, 256, 256)
        SSIM=tf.image.ssim(generated_image,image,max_val=255).numpy()
        SSIMs.append(SSIM)
    return max(SSIMs)

def main():
    # Check the image in the file path
    names=os.listdir('./keras_png_slices_data/keras_png_slices_train/')
    names=np.array(names,dtype=object)
    # Load data
    BATCH_SIZE = 256
    name_ds = tf.data.Dataset.from_tensor_slices(names)
    image_ds = name_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Batch and shuffle the data
    image_ds = image_ds.shuffle(buffer_size=2000).batch(BATCH_SIZE)
    # Create the models
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # Set optimizer
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    # Save checkpoints
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    
    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16

    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    # Train the model
    train(image_ds, EPOCHS)
    # Restore the latest checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # Generate an image with the trained generator model
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    generated_image=generated_image[0, :, :, :]* 127.5 + 127.5
    # Calculate SSIM
    print(max_ssim(generated_image))

if __name__ == "__main__":
    main()
