#!/usr/bin/env python
# coding: utf-8

# # Autoencoders
# 
# Autoencoder in TF Keras with the OASIS dataset

# In[3]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
import numpy as np
print(tf.config.list_physical_devices('GPU'))


# Parameters for the network

# In[4]:


depth = 32 #how much encoding to apply, compression of factor 24.5 if MNIST 28*28
length = 256*256
batch_size = 32

original_dim = 256*256
latent_dim = 16


# In[14]:


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

# In[38]:


#load the data

train, validate= load_data("C:\COMP3710\AKOA_Analysis")

# turing data into 1dim arrays without the labels        
# x_train = np.concatenate([x for x, y in train], axis=0)
# x_validate = np.concatenate([x for x, y in validate], axis=0)

x_train = tf.concat([x for x in train], axis=0)
x_validate = tf.concat([x for x in validate], axis=0)

x_train = x_train / 255.
x_validate = x_validate / 255.

print(x_train.shape)
print(x_validate.shape)


# In[35]:


# mean of the data
mean, var = tf.nn.moments(x_train, axes=[1])


# In[ ]:


# class GAN(keras.Model):
#     def __init__(self, self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
#         super(GAN, self).__init__()
#         self._embedding_dim = embedding_dim
#         self._num_embeddings = num_embeddings
        
#         self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
#         self._embedding.weight.data.normal_()
#         self._commitment_cost = commitment_cost
        
#         self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
#         self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
#         self._ema_w.data.normal_()
        
#         self._decay = decay
#         self._epsilon = epsilon

#     def compile(self, d_optimizer, g_optimizer, loss_fn):
#         super(GAN, self).compile()
#         self.d_optimizer = d_optimizer
#         self.g_optimizer = g_optimizer
#         self.loss_fn = loss_fn
#         self.d_loss_metric = keras.metrics.Mean(name="d_loss")
#         self.g_loss_metric = keras.metrics.Mean(name="g_loss")

#     @property
#     def metrics(self):
#         return [self.d_loss_metric, self.g_loss_metric]

#     def train_step(self, real_images):
#         # Sample random points in the latent space
#         batch_size = tf.shape(real_images)[0]
#         random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

#         # Decode them to fake images
#         generated_images = self.generator(random_latent_vectors)

#         # Combine them with real images
#         combined_images = tf.concat([generated_images, real_images], axis=0)

#         # Assemble labels discriminating real from fake images
#         labels = tf.concat(
#             [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
#         )
#         # Add random noise to the labels - important trick!
#         labels += 0.05 * tf.random.uniform(tf.shape(labels))

#         # Train the discriminator
#         with tf.GradientTape() as tape:
#             predictions = self.discriminator(combined_images)
#             d_loss = self.loss_fn(labels, predictions)
#         grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
#         self.d_optimizer.apply_gradients(
#             zip(grads, self.discriminator.trainable_weights)
#         )

#         # Sample random points in the latent space
#         random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

#         # Assemble labels that say "all real images"
#         misleading_labels = tf.zeros((batch_size, 1))

#         # Train the generator (note that we should *not* update the weights
#         # of the discriminator)!
#         with tf.GradientTape() as tape:
#             predictions = self.discriminator(self.generator(random_latent_vectors))
#             g_loss = self.loss_fn(misleading_labels, predictions)
#         grads = tape.gradient(g_loss, self.generator.trainable_weights)
#         self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

#         # Update metrics
#         self.d_loss_metric.update_state(d_loss)
#         self.g_loss_metric.update_state(g_loss)
#         return {
#             "d_loss": self.d_loss_metric.result(),
#             "g_loss": self.g_loss_metric.result(),
#         }


# In[53]:


# def relu_bn(inputs: tf.Tensor) -> tf.Tensor:
#     relu = ReLU()(inputs)
#     bn = BatchNormalization()(relu)
#     return bn

# VectorQuantizer = keras.Sequential(
#     [
#         keras.Input(shape=(128, 128, 3)),
#         layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
#         layers.ReLU(),
#         layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
#         layers.ReLU(),
#         layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
#         layers.ReLU(),
#     ],
#     name="VectorQuantizer",
# )
# VectorQuantizer.summary()


# In[10]:


#Images encdoer
Encoder = keras.Sequential(
    [
        keras.Input(shape=(128, 128, 1)),
        layers.Conv2D(32, 3, strides=2, padding="same"),
        layers.ReLU(),
        layers.Conv2D(64, 3, strides=2, padding="same"),
        layers.ReLU(),
        layers.Conv2D(latent_dim, 1, strides=2, padding="same"),
    ],
    name="Encoder",
)
Encoder.summary()


# In[11]:


#Images decoder
Decoder = keras.Sequential(
    [
        keras.Input(shape=Encoder.output.shape[1:]),
        layers.Conv2DTranspose(64, 3, strides=2, padding="same"),
        layers.ReLU(),
        layers.Conv2DTranspose(32, 3, strides=2, padding="same"),
        layers.ReLU(),
        layers.Conv2DTranspose(1, 3, strides=2, padding="same"),
    ],
    name="Decoder",
)
Decoder.summary()

