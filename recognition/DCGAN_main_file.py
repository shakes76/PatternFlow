#!/usr/bin/env python
# coding: utf-8

# In[1]:

#author : shivajiparala 
#student.no: 45512746
#libraries need to be installed 



import tensorflow as tf
import random
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import os, time 
from skimage import color
from skimage import io
import imageio
import PIL
import cv2 
import glob 
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.backend.tensorflow_backend import set_session
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# In[3]:


# lets how many gpu are available
tf.__version__
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[4]:


from functions import image_data


# In[5]:


# configuring the usage of gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
config.gpu_options.visible_device_list = "0" 
set_session(tf.compat.v1.Session(config=config))   


# In[6]:

# get the data
X_seg_train = image_data('keras_png_slices_train')


# polt traning images
fig, axs = plt.subplots(2, 2,figsize=(9,9))
axs[0, 0].imshow(X_seg_train[1],cmap="gray")
axs[0, 1].imshow(X_seg_train[2],cmap="gray")
axs[1, 0].imshow(X_seg_train[3],cmap="gray")
axs[1, 1].imshow(X_seg_train[4],cmap="gray")



# Normalize the images to [-1, 1]
train_images = (X_seg_train - 127.5) / 127.5 


# setting the batch size and buffer size 
BUFFER_SIZE = 8000
BATCH_SIZE = 128


# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


from functions import make_generator_model


# generator model generate image with noise before traning

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

# model architeture
generator.summary()

from functions import make_discriminator_model
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

from functions import generator_loss,discriminator_loss

# cross entropy function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)



generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 20


from functions import train_step,train
from IPython import display
from functions import generate_and_save_images


#traning the model
(g,d) = train(train_dataset,100,generator=generator,discriminator=discriminator,generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer)


#loss plots
plt.plot(g,label="generator_loss")
plt.plot(d,label = "discriminator_loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss_values")


# plot generated images from generator
fig, axs = plt.subplots(2, 2,figsize=(15,15))
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
axs[0, 0].imshow(generated_image[0, :, :, 0], cmap='gray')
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
axs[0, 1].imshow(generated_image[0, :, :, 0], cmap='gray')
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
axs[1, 0].imshow(generated_image[0, :, :, 0], cmap='gray')
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
axs[1, 1].imshow(generated_image[0, :, :, 0], cmap='gray')





