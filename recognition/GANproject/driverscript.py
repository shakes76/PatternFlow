#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *

import numpy as np

from matplotlib import *
from matplotlib import pyplot
from matplotlib.pyplot import *

from PIL import Image

import glob 
import os
import sys

from skimage import metrics
from skimage.metrics import structural_similarity as ssim

from modelscript import define_generator, define_discriminator

#check if GPU is available
tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#check current directory and import images
print(os.getcwd()) 
filelist=glob.glob('H:/COMP3710/keras_png_slices_train/*.png')
train_size = len(filelist)
images0=np.array([np.array(Image.open(i),dtype="float32") for i in filelist[0:train_size]])
print('training images',images0.shape)

filelist=glob.glob('H:/COMP3710/keras_png_slices_test/*.png')
test_size = len(filelist)
images1=np.array([np.array(Image.open(i),dtype="float32") for i in filelist[0:test_size]])
print('test images',images1.shape)

filelist=glob.glob('H:/COMP3710/keras_png_slices_validate/*.png')
val_size = len(filelist)
images2=np.array([np.array(Image.open(i),dtype="float32") for i in filelist[0:val_size]])
print('validation images',images2.shape)

#concatenate all images into one array called "images"
images=np.concatenate((images0,images1,images2), axis=0)
print(images.shape)

#######################################################################
#Preprocessing
#normalise pixel values from [0,255] to [-1,1]
images=(images - 127.5) / 127.5

#make into 4D array
images=images[:,:,:,np.newaxis]

#check shape
print(images.shape)

#######################################################################
#Check the brains, plot the first 10
pyplot.figure(figsize=(25,25))
for i in range(10):
    # define subplot
    pyplot.subplot(5, 5, 1 + i)
    # turn off axis
    pyplot.axis('off')
    # plot raw pixel data
    pyplot.imshow(images[i,:,:,0],cmap="gray")
pyplot.show()

########################################################################
#Call the generator and discriminator models

g_model = define_generator()

d_model = define_discriminator()

########Visualising generated images############
#choose the number of samples to visualise
n_samples=5
#define number of points in latent space
latent_dim=256


#generate noise according to number of samples specified with latent_dim previously defined as 256
noise = tf.random.normal([n_samples, latent_dim])

#generate fake images
x_fake = g_model(noise,training=False)
pyplot.figure(figsize=(25,25))
for i in range(n_samples):
    # define subplot
    pyplot.subplot(5, 5, 1 + i)
    pyplot.axis('off')
    # plot single image
    pyplot.imshow(x_fake[i, :, :,0],cmap='gray')
pyplot.show()
pyplot.close()

########################################################################
#Define loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

########################################################################
#Define optimisers

generator_optimiser = tf.keras.optimizers.Adam(lr=0.0002)

discriminator_optimiser = tf.keras.optimizers.Adam(lr=0.0001)

########################################################################
#Define training function
batch_size = 10

#Training function
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = g_model(noise, training=True)

        real_output = d_model(images, training=True)
        fake_output = d_model(generated_images, training=True)

        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(g_loss, g_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(d_loss, d_model.trainable_variables)

    generator_optimiser.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))
    discriminator_optimiser.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))
    return d_loss, g_loss

#########################################################################
#Define training loop
EPOCHS = 200
batch_per_epoch=np.round(images.shape[0]/batch_size)

#number of sample images to display
n_samples=5

total_size=images.shape[0]


# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(total_size).batch(batch_size)

def train(dataset, epochs):
    
    for epoch in range(epochs):
        count=0
        for image_batch in dataset:
            d_loss,g_loss=train_step(image_batch)
            if (count) % 25 == 0:
                print('>%d, %d/%d, d=%.8f, g=%.8f' % (epoch, count, batch_per_epoch, d_loss, g_loss))
            if (count) % 350 == 0:
                noise = tf.random.normal([n_samples, latent_dim])
                x_fake = g_model(noise,training=False)
                pyplot.figure(figsize=(25,25))
                for i in range(n_samples):
                    # define subplot
                    pyplot.subplot(5, 5, 1 + i)
                    pyplot.axis('off')
                    # plot single image
                    pyplot.imshow(x_fake[i, :, :,0],cmap='gray')
                pyplot.savefig('0511 Epoch{0} batch{1}.png'.format(epoch,count))
                pyplot.show()
                
                pyplot.close()
                #just save one model version per epoch
                filename = 'generator_model_%03d.h5' % (epoch) 
                g_model.save(filename)
            count=count+1

train(train_dataset, EPOCHS)

#######look at generator images########
n_samples=5
noise = tf.random.normal([n_samples, latent_dim])
x_fake = g_model(noise,training=False)

pyplot.figure(figsize=(25,25))
for i in range(n_samples):
    # define subplot
    pyplot.subplot(5, 5, 1 + i)
    pyplot.axis('off')
    # plot single image
    pyplot.imshow(x_fake[i, :, :,0],cmap='gray')
pyplot.show()
pyplot.close()

############SSIM#################
#since calculating SSIM for one image is computationally expensive, just choose the index of one image to calculate
#whichfake is the index of the sample image
whichfake=4

#create array to store SSIM values
ssim_noise=[]

#calculate SSIM for each training image
for i in range(images.shape[0]):
    ssim_noise.append( ssim(images[i,:,:,0], x_fake.numpy()[whichfake,:,:,0], 
                      data_range=np.max(x_fake.numpy()[whichfake,:,:,0]) - np.min(x_fake.numpy()[whichfake,:,:,0])))

#plot generated image and OASIS image that corresponds to the highest SSIM value
fig, axs = pyplot.subplots(2, 1, constrained_layout=True,figsize=(10,10))
axs[0].imshow(x_fake[whichfake, :, :, 0],cmap="gray")
axs[0].set_title('Generated image with max SSIM: {:.4f}'.format(np.max(ssim_noise)))

axs[1].imshow(images[ssim_noise.index(np.max(ssim_noise)), :, :, 0],cmap="gray")
axs[1].set_title('Closest OASIS image {:.0f}'.format(ssim_noise.index(np.max(ssim_noise))))

pyplot.show()

