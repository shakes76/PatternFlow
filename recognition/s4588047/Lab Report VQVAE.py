#!/usr/bin/env python
# coding: utf-8

# # Autoencoders
# 
# Autoencoder in TF Keras with the OASIS dataset

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from keras import backend as K
from tensorflow.keras.losses import binary_crossentropy, mse
from matplotlib import pyplot as plt
import numpy as np
print(tf.config.list_physical_devices('GPU'))


# Parameters for the network

# In[2]:


depth = 32 #how much encoding to apply, compression of factor 24.5 if MNIST 28*28
length = 256*256
batch_size = 16

original_dim = 256*256
latent_dim = 128


# In[3]:


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

# In[4]:


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


# In[5]:


#test images

plt.imshow(x_train[1])
plt.show()


# In[6]:


# mean of the data
mean, var = tf.nn.moments(x_train, axes=[1])



 class VectorQuantizer(layers.Layer):  
     def __init__(self, embedding_dim, num_embeddings, commitment_cost=0.25, epsilon=1e-10, **kwargs):
         self.embedding_dim = embedding_dim
         self.num_embeddings = num_embeddings
         self.commitment_cost = commitment_cost
         super(VectorQuantizer, self).__init__(**kwargs)

     def build(self, input_shape):
         # Add embedding weights.
         self.w = self.add_weight(name='embedding',
                                   shape=(self.embedding_dim, self.num_embeddings),
                                   initializer='uniform',
                                   trainable=True)
         # Finalize building.
         super(VectorQuantizer, self).build(input_shape)

     def call(self, x):
         # Covert into flatten representation (eg. (14944, 128, 128, 3) to (x, y))
         flat_inputs = K.reshape(x, (-1, self.embedding_dim))

         # Calculate distances of input by calcuating the distance of every encoded vector to the embedding space.
         distances = (K.sum(flat_inputs**2, axis=1, keepdims=True)
                      - 2 * K.dot(flat_inputs, self.w)
                      + K.sum(self.w ** 2, axis=0, keepdims=True))

         # Retrieve encoding indices.
         encoding_indices = K.argmax(-distances, axis=1)
         encodings = K.one_hot(encoding_indices, self.num_embeddings)
         encoding_indices = K.reshape(encoding_indices, K.shape(x)[:-1])
         quantized = self.quantize(encoding_indices)

         # Metrics.
         #avg_probs = K.mean(encodings, axis=0)
         #perplexity = K.exp(- K.sum(avg_probs * K.log(avg_probs + epsilon)))
        
         commitment_loss = 1 * tf.reduce_mean(
              (tf.stop_gradient(quantized) - x) ** 2)
         codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
         self.add_loss(commitment_loss + codebook_loss)
        
         return x + K.stop_gradient(quantized- x)

     @property
     def embeddings(self):
         return self.w

     def quantize(self, encoding_indices):
         w = K.transpose(self.embeddings.read_value())
         return tf.nn.embedding_lookup(w, encoding_indices)
    


# In[11]:


# # Calculate vq-vae loss.
# def vq_vae_loss_wrapper(data_variance, commitment_cost, quantized, x_inputs):
#     def vq_vae_loss(x, x_hat):
#         print(x.shape)
#         print(x_hat.shape)
#         recon_loss = mse(x, x_hat)
        
#         e_latent_loss = K.mean((K.stop_gradient(quantized) - x_inputs) ** 2)
#         q_latent_loss = K.mean((quantized - K.stop_gradient(x_inputs)) ** 2)
#         loss = q_latent_loss + commitment_cost * e_latent_loss
        
#         return recon_loss + loss #* beta
#     return vq_vae_loss

# class ScaleLayer(tf.keras.layers.Layer):
#     def __init__(self):
#       super(ScaleLayer, self).__init__()

#     def call(self, enc, enc_inputs):
#       return enc_inputs + K.stop_gradient(enc - enc_inputs)


# In[12]:


# # Encoder
# input_img = layers.Input(shape=(128, 128, 3))
# x = Encoder(input_img)

# # VQVAELayer.
# enc = layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), name="pre_vqvae")(x)
# enc_inputs = enc
# enc = VectorQuantizer(3, latent_dim, name="vqvae")(enc)
# #x = layers.Lambda(lambda enc: enc_inputs + K.stop_gradient(enc - enc_inputs), name="encoded")(enc)
# x = ScaleLayer()(enc, enc_inputs)

# loss = vq_vae_loss_wrapper(var, 0.25, enc, enc_inputs)

# # Decoder.
# x = Decoder(x)

# # Autoencoder.
# vqvae = Model(input_img, x)
# vqvae.compile(loss=loss, optimizer='adam')
# vqvae.summary()


# In[177]:


#Images encdoer
Encoder = keras.Sequential(
    [
        keras.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
        #layers.ReLU(),
        layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
        #layers.ReLU(),
        layers.Conv2D(latent_dim, 3, strides=2, padding="same", activation="relu"),
    ],
    name="Encoder",
)
Encoder.summary()

#Images decoder
VQ = keras.Sequential(
    [
        keras.Input(shape=Encoder.output.shape[1:]),
        VectorQuantizer(128, latent_dim),
    ],
    name="VQ",
)
VQ.summary()

#Images decoder
Decoder = keras.Sequential(
    [
        keras.Input(shape=VQ.output.shape[1:]),
        layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu"),
        #layers.ReLU(),
        layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu"),
        #layers.ReLU(),
        layers.Conv2DTranspose(3, 3, strides=2, padding="same"),
    ],
    name="Decoder",
)
Decoder.summary()

# #Images decoder
# VQVAE = keras.Sequential(
#     [
#         keras.Input(shape=(128, 128, 3)),
#         Encoder,
#         VectorQuantizer(128, latent_dim),
#         Decoder,
        
#     ],
#     name="VQVAE",
# )
# VQVAE.summary()

#Images decoder
VQVAE = keras.Sequential(
    [
        keras.Input(shape=(128, 128, 3)),
        Encoder,
        VQ,
        Decoder,
        
    ],
    name="VQVAE",
)
VQVAE.summary()


# In[212]:


e = Encoder
v = VectorQuantizer(128, latent_dim)
d = Decoder
optimizer = tf.keras.optimizers.Adam()

class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.

            e_out = e(x)
    #         print(e_out)
            v_out = v(e_out)
    #         print(v_out)

            r = d(v_out)
            lr = tf.keras.losses.MSE(x, r) + sum(v.losses)
            #print(tf.reduce_mean(lr))

        grads = tape.gradient(lr, v.trainable_variables)
        optimizer.apply_gradients(zip(grads, v.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(lr)

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
        }
    def call(self, inputs):
        print(inputs.shape)
        e_out = e(inputs)
        v_out = v(e_out)
        r = d(v_out)
        
        return r


# In[66]:


print(x_train.shape)


# In[176]:



# e = Encoder
# v = VectorQuantizer(128, latent_dim)
# d = Decoder
# optimizer = tf.keras.optimizers.Adam()

# for i in range(x_train.shape[0]):
#     x = x_train[i: i+1]
#     with tf.GradientTape() as tape:
#         # Outputs from the VQ-VAE.

#         e_out = e(x)
# #         print(e_out)
#         v_out = v(e_out)
# #         print(v_out)

#         r = d(v_out)
#         lr = tf.keras.losses.MSE(x, r) + sum(v.losses)
#         print(tf.reduce_mean(lr))
        
#     grads = tape.gradient(lr, v.trainable_variables)
#     optimizer.apply_gradients(zip(grads, v.trainable_variables))


# In[217]:


vqvae_trainer = VQVAETrainer(var)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
history = vqvae_trainer.fit(x_train, epochs=1, batch_size=batch_size)


# history = vqvae.fit(x_train, x_train, epochs=3, batch_size=batch_size)


# In[218]:


def show_subplot(original, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    
    plt.imshow(tf.cast(reconstructed * 255.,'uint8'))
    plt.title("Reconstructed")
    plt.axis("off")
    
    plt.show()
    
    im1 = tf.image.convert_image_dtype(original, tf.float32)
    im2 = tf.image.convert_image_dtype(tf.cast(reconstructed * 255.,'uint8'), tf.float32)
    ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03)
    print(ssim2)
    

xtrain = tf.expand_dims(x_train, axis=1)
print(xtrain[1].shape)


#trained_vqvae_model = vqvae_trainer
# e_out = e(tf.reshape(x_train[1000], (1,128,128,3)))
# v_out = v(e_out)
# r = d(v_out)

# recon = r

recon = vqvae_trainer.predict(tf.reshape(x_validate[1000], (1,128,128,3)))

print((recon).shape)
# for test_image in test_images:
#     show_subplot(test_image)

show_subplot(x_validate[1000], tf.squeeze(recon))


# In[203]:


# Plot training results.
loss = history.history['loss'] # Training loss.
num_epochs = range(1, 1 + len(history.history['loss'])) # Number of training epochs.

plt.figure(figsize=(16,9))
plt.plot(num_epochs, loss, label='Training loss') # Plot training loss.

plt.title('Training loss')
plt.legend(loc='best')
plt.show()

