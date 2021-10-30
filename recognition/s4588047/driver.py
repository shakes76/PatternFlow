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
import tensorflow_probability as tfp

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
            recon_error = tf.reduce_mean((x_recon - x) ** 2) #/ self._data_variance
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
history = vqvae_trainer.fit(x_train, epochs=20, batch_size=batch_size, validation_data=(x_validate,))
#%%


def show_subplot(original, reconstructed, i):
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
    #plt.savefig('outputs/sample' + str(i) + '.png')
    
    im1 = tf.image.convert_image_dtype(original, tf.float32)
    im2 = tf.image.convert_image_dtype(tf.cast(reconstructed * 255.,'uint8'), tf.float32)
    ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03)
    print(ssim2)
    
xtrain = tf.expand_dims(x_validate, axis=1)

print(xtrain[1000].shape)

images = []
for i in range(10):
    recon = vqvae_trainer.predict(xtrain[i])
    images.append(recon)

print((recon).shape)
# for test_image in test_images:
#     show_subplot(test_image)

#show_subplot(x_validate[1000], tf.squeeze(recon))
for i in range(10):
    show_subplot(x_validate[i], images[i][0], i)
#%%


# training history
# Plot training results.
loss = history.history['loss'] # Training loss.
recon_loss = history.history['reconstruction_loss']
vq_loss = history.history['vqvae_loss']             
num_epochs = range(1, 1 + len(history.history['loss'])) # Number of training epochs.

plt.figure(figsize=(16,9))
plt.plot(num_epochs, loss, label='Total loss') # Plot training loss.
plt.plot(num_epochs, recon_loss, label='Reconstruction loss') # Plot training loss.
plt.plot(num_epochs, vq_loss, label='VQ-VAE loss') # Plot training loss.

plt.title('Training loss')
plt.legend(loc='best')
plt.show()
#%%

tfd = tfp.distributions
from pixel_cnn import PixelConvLayer, ResidualBlock

dist = tfd.PixelCNN(
    image_shape=(128,128,1),
    num_resnet=1,
    num_hierarchies=2,
    num_filters=32,
    num_logistic_mix=5,
    dropout_p=.3,
)

encoded_outputs = encoder(xtrain[1:10])
flat_enc_outputs = tf.reshape(encoded_outputs, (-1, encoded_outputs.shape[-1]))
codebook_indices = vq_vae.get_code_indices(flat_enc_outputs)
codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
print("codes", codebook_indices.shape)
codebook_indices = tf.squeeze(codebook_indices)

for i in range(1):
    plt.subplot(1, 2, 1)
    plt.imshow(tf.squeeze(xtrain[i]) + 0.5)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(codebook_indices[i])
    plt.title("Code")
    plt.axis("off")
    plt.show()
    
num_residual_blocks = 2
num_pixelcnn_layers = 2
pixelcnn_input_shape = encoded_outputs.shape[1:-1]
print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")



pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
ohe = tf.one_hot(pixelcnn_inputs, num_embeddings)
x = PixelConvLayer(
    mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
)(ohe)

for _ in range(num_residual_blocks):
    x = ResidualBlock(filters=128)(x)

for _ in range(num_pixelcnn_layers):
    x = PixelConvLayer(
        mask_type="B",
        filters=128,
        kernel_size=1,
        strides=1,
        activation="relu",
        padding="valid",
    )(x)

out = keras.layers.Conv2D(
    filters=num_embeddings, kernel_size=1, strides=1, padding="valid"
)(x)

pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")
pixel_cnn.summary()


# Generate the codebook indices.
encoded_outputs = encoder(xtrain[1:10])
flat_enc_outputs = tf.reshape(encoded_outputs, (-1, encoded_outputs.shape[-1]))
codebook_indices = vq_vae.get_code_indices(flat_enc_outputs)

codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")

pixel_cnn.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
pixel_cnn.fit(
    x=codebook_indices,
    y=codebook_indices,
    batch_size=128,
    epochs=30,
    validation_split=0.1,
)

print(pixel_cnn.input_shape[1:])
# Create a mini sampler model.
inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
x = pixel_cnn(inputs, training=False)
dist = tfp.distributions.Categorical(logits=x)
sampled = dist.sample()
sampler = keras.Model(inputs, sampled)


# Create an empty array of priors.
batch = 10
priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
print(priors.shape)
_,batch, rows, cols = priors.shape

# # Iterate over the priors because generation has to be done sequentially pixel by pixel.
# for row in range(rows):
#     for col in range(cols):
#         # Feed the whole array and retrieving the pixel value probabilities for the next
#         # pixel.
#         first = pixel_cnn.predict(priors)
#         second = tfp.distributions.Categorical(first)
#         third = second.sample()
#         probs = third
        
#         #probs = sampler.predict(priors)
#         # Use the probabilities to pick pixel values and append the values to the priors.
#         priors[:, row, col] = probs[:, row, col]

print(f"Prior shape: {priors.shape}")



# Perform an embedding lookup.
pretrained_embeddings = vq_vae.embeddings
priors_ohe = tf.one_hot(priors.astype("int32"), num_embeddings).numpy()
quantized = tf.matmul(
    priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
)
quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))

# Generate novel images.

generated_samples = decoder(quantized[1])

for i in range(batch):
    plt.subplot(1, 2, 1)
    plt.imshow(priors[i])
    plt.title("Code")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(generated_samples[i].squeeze() + 0.5)
    plt.title("Generated Sample")
    plt.axis("off")
    plt.show()
