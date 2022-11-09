#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time, os
import matplotlib.pyplot as plt
## optimizer
#optimizer = Adam(0.0002, 0.5)
optimizer = Adam(0.00007, 0.5)
##optimizer = 'adam'



import tensorflow as tf
print(tf.__version__)
import glob
import cv2
from IPython import display


import matplotlib.pyplot as plt
import os, time  
import numpy as np 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


# In[2]:



def make_generator_model(input_shp = (100,), final_activation = 'tanh' ):
    """
    Function to create generator model w.r.t a latent input noise.
    Default noise dimensions: (100,)
    Default final activation: tanh
    
    """
    import numpy as np
    from tensorflow.keras import layers, models
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(0.00007, 0.5)
    
    model = tf.keras.Sequential()
    model.add(layers.Dense(16*16*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 256)))
    assert model.output_shape == (None, 16, 16, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(4, 4), padding='same', use_bias=False, activation=final_activation))
    assert model.output_shape == (None, 128, 128, 1)
    
    return model


# In[ ]:





# In[3]:


def make_discriminator_model(input_shp = [128, 128, 1]):
    """
    Function to create discriminator model w.r.t
    input image shape generated from generator.
    Default inputshape =  [128, 128, 1] 
    
    """
    
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=input_shp))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# In[4]:


def discriminator_loss(real_output, fake_output):
    
    """
    Function to calculate Loss function for Discriminator.
    core function for loss: tf.keras.losses.BinaryCrossentropy(from_logits=True)
    Returns total loss: (generated_image loss + real_image_loss)
    
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# In[5]:


def generator_loss(fake_output):
    """
    Function to calculate Loss function for Generator.
    core function for loss: tf.keras.losses.BinaryCrossentropy(from_logits=True)
    Returns total loss of generated image generated_image loss
    
    """
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# In[7]:


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(generator, discriminator, images, BATCH_SIZE = 128, noise_dim = 100):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    #tf.config.run_functions_eagerly(True)
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
      

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)	

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return  (gen_loss, disc_loss)
    


# In[8]:


# define Structure similarity function

def get_max_ssim(input_data, test):
    """
    Function to get Max SSIM for input data and test image.
    
    Core function: tf.image.ssim
    
    """
    
    ssim_val = 0
    
    
    for i in range(len(input_data)):
        s = tf.image.ssim(input_data[i], test,1).numpy()
        if (s > ssim_val):
            ssim_val = s
        #if ssim_val > 0.6:
        #    break
    return ssim_val


# In[9]:


def generate_and_save_images(model, epoch, test_input, plotsize = (8,8), save_epoch = 5):
    
    """
    Image helper function to generate and save images.
    
    This function is called with every train step to visualise the generated images with each epoch. 
    
    
    """
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=plotsize)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    gen_images_dir = './gen_images'
    gen_images_prefix = os.path.join(gen_images_dir, "image_at_epoch_")

    if(epoch % save_epoch ==0):
        plt.savefig(gen_images_prefix + '{:04d}.png'.format(epoch))
    plt.show()


# In[10]:



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
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)
        
        
        gen_losses.append(gen_loss.numpy())
        disc_losses.append(disc_loss.numpy())
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        
        
        test_img = tf.image.convert_image_dtype(generator.predict(np.asarray(seed).reshape(16,100))[0], tf.float32, saturate=False, name=None)
        ssim = get_max_ssim(tf_X_train[:500],test_img)
        ssims.append(ssim)                    
        print('ssim = ', ssim)
        


    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                           epochs,
                           seed)
    
    return [ssims, gen_losses, disc_losses]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:

#
#generator_optimizer = tf.keras.optimizers.Adam(1e-4)
#discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
#
#
## In[ ]:
#
#
## Test scripts for DCGAN_utilities
#
#
## In[11]:
#
#
#import image_helper_for_DCGAN as helper
#
#data_dir = "H://PatternLab//PatternRecognition//Datasets//brain//keras_png_slices_data//keras_png_slices_train//path_test"
#tf_X_train = helper.cv_get_images(data_dir)
#tf_X_train.shape
#
#print('max = ', tf.reduce_max(tf_X_train).numpy())
#print('min = ', tf.reduce_min(tf_X_train).numpy())
#
#
## In[13]:
#
#
#BUFFER_SIZE = 8000
#BATCH_SIZE = 256
#
## Batch and shuffle the data
#train_dataset = tf.data.Dataset.from_tensor_slices(tf_X_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#train_dataset
#
#
## In[ ]:
#
#
##Create a Train_batch
#
#
## In[14]:
#
#
##create a generator
#
#generator = make_generator_model()
#generator.summary()
#noise = tf.random.normal([1, 100])
#generated_image = generator(noise, training=False)
#
#plt.imshow(generated_image[0, :, :, 0], cmap='gray')
#
#
## In[15]:
#
#
## Create a Discriminator
#
#discriminator = make_discriminator_model()
#discriminator.summary()
#decision = discriminator(generated_image)
#print (decision)
#
#
## In[17]:
#
#
##defining Optimisers:
#
#generator_optimizer = tf.keras.optimizers.Adam(1e-4)
#discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
#
#
## In[19]:
#
#
## Defnine Training Loop
#
#EPOCHS = 5
#noise_dim = 100
#num_examples_to_generate = 16
#
## We will reuse this seed overtime (so it's easier)
## to visualize progress in the animated GIF)
#seed = tf.random.normal([num_examples_to_generate, noise_dim])
#
#
## In[22]:
#
#
#[ssims, gen_losses, disc_losses] = train(train_dataset, EPOCHS)
#
#
## In[24]:
#
#
#plt.plot(ssims)
#plt.plot(gen_losses)
#plt.plot(disc_losses)
#
#
## In[25]:
#
#
#import imageio
#anim_file = 'dcgan.gif'
#
#with imageio.get_writer(anim_file, mode='I') as writer:
#  filenames = glob.glob('./gen_images/image*.png')
#  filenames = sorted(filenames)
#  for filename in filenames:
#    image = imageio.imread(filename)
#    writer.append_data(image)
#  image = imageio.imread(filename)
#  writer.append_data(image)
#
#
## In[29]:
#
#
#plt.figure(figsize=(10,10))
#import tensorflow_docs.vis.embed as embed
#embed.embed_file(anim_file)
#
#
## In[ ]:
#
#
#

