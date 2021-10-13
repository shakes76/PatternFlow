#!/usr/bin/env python
# coding: utf-8

# # Autoencoders
# 
# Autoencoder in TF Keras with the OASIS dataset

# In[22]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from matplotlib import pyplot as plt
import numpy as np
print(tf.config.list_physical_devices('GPU'))


# Parameters for the network

# In[23]:


depth = 32 #how much encoding to apply, compression of factor 24.5 if MNIST 28*28
length = 256*256
batch_size = 32

original_dim = 256*256
latent_dim = 16


# In[24]:


def load_data(path):
    # Returns images dataset of each data catagory
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        label_mode=None,
        image_size=(128, 128),
        batch_size=batch_size,
        subset='training',
        validation_split=0.2,
        seed = 123
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        label_mode=None,
        image_size=(128, 128),
        batch_size=batch_size,
        subset='validation',
        validation_split=0.2,
        seed = 123
    )
    return train_ds, val_ds


# Load the dataset, reshape as an image and normalise it

# In[25]:


#load the data

train, validate= load_data("C:\COMP3710\AKOA_Analysis")

# turing data into 1dim arrays without the labels        
# x_train = np.concatenate([x for x, y in train], axis=0)
# x_validate = np.concatenate([x for x, y in validate], axis=0)

x_train = tf.concat([x for x in train], axis=0)
x_validate = tf.concat([x for x in validate], axis=0)

x_train = x_train / 255.
x_test = x_validate / 255.

print(x_train.shape)
print(x_validate.shape)


# In[26]:


#test images

plt.imshow(x_train[1])
plt.show()


# In[27]:


# mean of the data
mean, var = tf.nn.moments(x_train, axes=[1])


# In[28]:


#Images encdoer
Encoder = keras.Sequential(
    [
        keras.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, 3, strides=2, padding="same"),
        layers.ReLU(),
        layers.Conv2D(64, 3, strides=2, padding="same"),
        layers.ReLU(),
        layers.Conv2D(latent_dim, 1, strides=2, padding="same"),
    ],
    name="Encoder",
)
Encoder.summary()


# In[29]:


#Images decoder
Decoder = keras.Sequential(
    [
        keras.Input(shape=Encoder.output.shape[1:]),
        layers.Conv2DTranspose(64, 3, strides=2, padding="same"),
        layers.ReLU(),
        layers.Conv2DTranspose(32, 3, strides=2, padding="same"),
        layers.ReLU(),
        layers.Conv2DTranspose(3, 3, strides=2, padding="same"),
    ],
    name="Decoder",
)
Decoder.summary()


# In[30]:


class VectorQuantizer(layers.Layer):
    pass


# In[31]:


#Images decoder
VQVAE = keras.Sequential(
    [
        keras.Input(shape=(128, 128, 3)),
        Encoder,
        VectorQuantizer(64, latent_dim),
        Decoder,
        
    ],
    name="VQVAE",
)
VQVAE.summary()


# In[32]:


class VQVAETrainer(keras.models.Model):
    pass


# In[ ]:


vqvae_trainer = VQVAETrainer(var, latent_dim=16, num_embeddings=128)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
vqvae_trainer.fit(x_train, epochs=5, batch_size=128)


# In[17]:


def show_subplot(original, reconstructed=None):
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed)
    plt.title("Reconstructed")
    plt.axis("off")
    
    plt.show()

xtrain = tf.expand_dims(x_train, axis=1)
print(xtrain[1].shape)


trained_vqvae_model = vqvae_trainer.vqvae

recon = trained_vqvae_model.predict(xtrain[1])
print(tf.squeeze(recon).shape)
# for test_image in test_images:
#     show_subplot(test_image)

show_subplot(x_train[1], tf.squeeze(recon))

