#!/usr/bin/env python
# coding: utf-8

# # Driver File for DCGAN Brain Image data.
# 
# ## Information available at: 
# 
# https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/DC_GAN_Readme.md

# In[11]:


# Importing dependencies

# Tensorflow utilities
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
print('TF version: ', tf.__version__)

#generic python utilities
from IPython import display
import matplotlib.pyplot as plt
import glob
import cv2
import os, time  

#helper utilities
import image_helper_for_DCGAN as helper
import DCGAN_utilities as DCGAN
import numpy as np


# ## Loading the data in Tensor format from local directory.

# In[2]:




data_dir = "H://PatternLab//PatternRecognition//Datasets//brain//keras_png_slices_data//keras_png_slices_train//"
tf_X_train = helper.cv_get_images(data_dir)
tf_X_train.shape

print('max = ', tf.reduce_max(tf_X_train).numpy())
print('min = ', tf.reduce_min(tf_X_train).numpy())


# ## Visualise the data using image helper functions
# 
# 

# In[3]:


helper.tf_plot_gallery(tf_X_train, (128,128,1))


# ## Create train batch generator that returns random batch size image input to be used as train data.
# 

# In[4]:


# Create train batch generator

BUFFER_SIZE = 8000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(tf_X_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset


# ## Initializing generator from DCGAN helper file

# In[5]:


# Initialise generator:

generator = DCGAN.make_generator_model()
generator.summary()


# In[6]:


noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')


# ## Initializing Discriminator from DCGAN helper file

# In[7]:


# Initialize discriminator

discriminator = DCGAN.make_discriminator_model()
discriminator.summary()
decision = discriminator(generated_image)
print (decision)


# ## Defnine Training Loop with looping parameters

# In[8]:



EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16

## We will reuse this seed overtime (so it's easier)
## to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# ## Defining training Functions
# 

# In[9]:





def train_step(generator, discriminator, images, BATCH_SIZE = 128, noise_dim = 100):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    #tf.config.run_functions_eagerly(True)
    # Defining optimizers for loss functions.
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = DCGAN.generator_loss(fake_output)
        disc_loss = DCGAN.discriminator_loss(real_output, fake_output)
      

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)	

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return  (gen_loss, disc_loss)
 

def train(generator, discriminator, dataset, epochs):
    """
    Main function to train the models.
    
    returns SSIM, Generator loss and Discriminator loss
    
    """
    
    ssims = []
    gen_losses = []
    disc_losses = []
    
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            (gen_loss, disc_loss) =  train_step(generator, discriminator, image_batch)
        
        # Produce images for the GIF as we go
        
        display.clear_output(wait=True)
        DCGAN.generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)
        
        
        gen_losses.append(gen_loss.numpy())
        disc_losses.append(disc_loss.numpy())
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        
        
        test_img = tf.image.convert_image_dtype(generator.predict(np.asarray(seed).reshape(16,100))[0], tf.float32, saturate=False, name=None)
        ssim = DCGAN.get_max_ssim(tf_X_train[:500],test_img)
        ssims.append(ssim)                    
        print('ssim = ', ssim)
        


    # Generate after the final epoch
    display.clear_output(wait=True)
    DCGAN.generate_and_save_images(generator,
                           epochs,
                           seed)
    
    return [ssims, gen_losses, disc_losses]


# ## Calling train Loop

# In[12]:


[ssims, gen_losses, disc_losses] = train(generator, discriminator, train_dataset, EPOCHS)


# In[27]:


#[ssims, gen_losses, disc_losses]


# ## Visualising the progress of the train Loops using plots

# In[26]:


fig = plt.figure(figsize=(10,6))
plt.plot(ssims, label = 'SSIM')
plt.title('Structure similarity for generated image.')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('SSIM')
plt.grid()
plt.show()


# In[15]:


fig = plt.figure(figsize=(10,6))
plt.plot(gen_losses, label = 'Generator Loss')
plt.plot(disc_losses, label = 'Discriminator Loss')
plt.legend()
plt.grid()
plt.title('Loss w.r.t Epochs')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.show()


# ## Visualising test images:

# In[21]:


# genrating random normal noise
test_noise = tf.random.normal([16, noise_dim])

#generating images from noise
test_gen_imgs = generator(test_noise)

#displaying the images
helper.tf_plot_gallery(test_gen_imgs, (128,128,1))


# In[24]:


# creating a gif for trained images

import imageio
anim_file = 'dcgan_20201104_1.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('./gen_images/image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)


# In[25]:


import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)


# In[ ]:




