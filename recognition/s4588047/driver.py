#!/usr/bin/env python
# coding: utf-8

# # Autoencoders
# 
# Autoencoder in TF Keras with the AKOA Knee dataset

#%%
# importing helper classes and functions
from data_loader import load_data
from model import Encoder, Decoder, VectorQuantizer

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
print(tf.config.list_physical_devices('GPU'))

#%%
# Parameters for the network
batch_size = 128
num_hiddens = 128
num_residual_hiddens = 64
num_residual_layers = 4

# the capacity in the information-bottleneck.
embedding_dim = 128
# The higher this value, the higher the capacity in the information bottleneck.
num_embeddings = 512
# the scale of the reconstruction cost (log p(x|z)).
commitment_cost = 0.25
# Learnning rate for the optimizer
learning_rate = 2e-4
#%%

# Load the dataset, reshape as an image and normalise it
x_train, x_validate= load_data(r"C:\Users\Yousif\Desktop\3710\AKOA_Analysis")


#test images
plt.imshow(x_train[1])
plt.show()
#%%

# mean of the data
# mean, var = tf.nn.moments(x_train, axes=[1])
data_variance = np.var(x_train / 255.0) # only works with np


# Build modules.
encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens)
pre_vq_conv1 = layers.Conv2D(embedding_dim, (1, 1), strides=(1, 1), name="to_vq")
vq_vae = VectorQuantizer(
  embedding_dim=embedding_dim,
  num_embeddings=num_embeddings,
  beta=commitment_cost)


#Images decoder for testing purposes
hh = keras.Sequential(
    [
        keras.Input(shape=(128, 128, 1)),
        encoder,
        pre_vq_conv1,
        vq_vae,
        decoder,
        
    ],
    name="VQVAE",
)
hh.summary()

class VQVAETrainer(keras.models.Model):
    def __init__(self, encoder, decoder, vq_vae, pre_vq_conv1, data_variance, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)

        self._encoder = encoder
        self._decoder = decoder
        self._vqvae = vq_vae
        self._pre_vq_conv1 = pre_vq_conv1
        self._data_variance = data_variance

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:

            #lr = tf.keras.losses.MSE(x, r) + sum(v.losses)
            
            z = self._pre_vq_conv1(self._encoder(x))
            print("_pre_vq_conv1", z.shape)
            vq_output = self._vqvae(z)
            print("vq_output", vq_output.shape)
            x_recon = self._decoder(vq_output)
            print("_decoder", x_recon.shape)
            recon_error = tf.reduce_mean((x_recon - x) ** 2) / self._data_variance
            loss = recon_error + sum(self._vqvae.losses)


        grads = tape.gradient(loss, self._vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self._vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(recon_error)
        self.vq_loss_tracker.update_state(sum(self._vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }
    def call(self, inputs):
        z = self._pre_vq_conv1(self._encoder(inputs))
        vq_output = self._vqvae(z)
        x_recon = self._decoder(vq_output)
        
        return x_recon


vqvae_trainer = VQVAETrainer(encoder, decoder, vq_vae, pre_vq_conv1, data_variance)
# using adam optimizer for the gradient training
vqvae_trainer.compile(optimizer=keras.optimizers.Adam(learning_rate))
history = vqvae_trainer.fit(x_train, epochs=1, batch_size=batch_size, validation_data=(x_validate,))
#%%


def show_subplot(original, reconstructed, s1=None):
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    
    plt.imshow(tf.cast(reconstructed * 255.,'uint8'))
    #plt.imshow(tf.cast(s1 * 255.,'uint8'))
    plt.title("Reconstructed")
    plt.axis("off")
    
    plt.show()
    
    im1 = tf.image.convert_image_dtype(original, tf.float32)
    im2 = tf.image.convert_image_dtype(tf.cast(reconstructed * 255.,'uint8'), tf.float32)
    ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03)
    print(ssim2)
    
xtrain = tf.expand_dims(x_validate, axis=1)

print(xtrain[1000].shape)
recon = vqvae_trainer.predict(xtrain[1000])

print((recon).shape)
# for test_image in test_images:
#     show_subplot(test_image)

#show_subplot(x_validate[1000], tf.squeeze(recon))
show_subplot(x_validate[1000], recon[0])
#%%


# training history
# Plot training results.
loss = history.history['loss'] # Training loss.
num_epochs = range(1, 1 + len(history.history['loss'])) # Number of training epochs.

plt.figure(figsize=(16,9))
plt.plot(num_epochs, loss, label='Training loss') # Plot training loss.

plt.title('Training loss')
plt.legend(loc='best')
plt.show()
