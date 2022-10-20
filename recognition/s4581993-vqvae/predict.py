import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp

from dataset import get_test_dataset, get_train_dataset
from modules import VQVAETrainer, get_pixelcnn
from utils import models_directory, vqvae_weights_filename, pixelcnn_weights_filename

# Make sure the trained weights exist
if not os.path.isfile(models_directory + vqvae_weights_filename + ".index"):
    print("Missing VQ-VAE training weights. Please run train.py", file=sys.stderr)
    exit(1)
if not os.path.isfile(models_directory + pixelcnn_weights_filename + ".index"):
    print("Missing PixelCNN training weights. Please run train.py", file=sys.stderr)
    exit(1)

# Load testing dataset
test_ds = get_test_dataset()

# Create the model and load the weights
train_ds = get_train_dataset()
data_variance = np.var(train_ds)
vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
vqvae_trainer.load_weights(models_directory + vqvae_weights_filename)

# Visualise results of encoding and decoding
test_ds = get_test_dataset()
trained_vqvae_model = vqvae_trainer.vqvae
idx = np.random.choice(len(test_ds), 4)
test_images = test_ds[idx]
reconstructions_test = trained_vqvae_model.predict(test_images)

encoder = vqvae_trainer.vqvae.get_layer("encoder")
quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

encoded_outputs = encoder.predict(test_images)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

plt.figure(figsize=(12, len(test_images) * 4))
plt.subplot(len(test_images), 3, 1)
plt.title("Original")
plt.subplot(len(test_images), 3, 2)
plt.title("Code")
plt.subplot(len(test_images), 3, 3)
plt.title("Decoded")

for i in range(len(test_images)):
    plt.subplot(len(test_images), 3, i * 3 + 1)
    plt.imshow(test_images[i].squeeze() + 0.5, cmap='gray')
    plt.axis("off")

    plt.subplot(len(test_images), 3, i * 3 + 2)
    plt.imshow(codebook_indices[i] + 0.5, cmap='gray')
    plt.axis("off")

    plt.subplot(len(test_images), 3, i * 3 + 3)
    plt.imshow(reconstructions_test[i].squeeze() + 0.5, cmap='gray')
    plt.axis("off")
plt.tight_layout()
plt.show()

# Load the PixelCNN model
num_residual_blocks = 2
num_pixelcnn_layers = 2

# Encode an image to get the output shape
# I'm sure there's a better way to do this, but the custom layers make it hard
encoder = vqvae_trainer.vqvae.get_layer("encoder")
quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")
encoded_output = encoder.predict(train_ds[np.newaxis, 0])
pixelcnn_input_shape = encoded_output.shape[1:-1]
print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")

pixel_cnn = get_pixelcnn(
        num_residual_blocks,
        num_pixelcnn_layers,
        pixelcnn_input_shape,
        vqvae_trainer.num_embeddings,
)
pixel_cnn.load_weights(models_directory + pixelcnn_weights_filename)

# Generate new images with the PixelCNN model
inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
outputs = pixel_cnn(inputs, training=False)
categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
outputs = categorical_layer(outputs)
sampler = tf.keras.Model(inputs, outputs)

# Create an empty array of priors.
batch = 4
priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
batch, rows, cols = priors.shape

# Iterate over the priors because generation has to be done sequentially pixel by pixel.
for row in range(rows):
    for col in range(cols):
        # Feed the whole array and retrieving the pixel value probabilities for the next
        # pixel.
        probs = sampler.predict(priors)
        # Use the probabilities to pick pixel values and append the values to the priors.
        priors[:, row, col] = probs[:, row, col]

print(f"Prior shape: {priors.shape}")

# Perform an embedding lookup.
pretrained_embeddings = quantizer.embeddings
priors_ohe = tf.one_hot(priors.astype("int32"), vqvae_trainer.num_embeddings).numpy()
quantized = tf.matmul(
    priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
)
quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))

# Generate novel images.
decoder = vqvae_trainer.vqvae.get_layer("decoder")
generated_samples = decoder.predict(quantized)

plt.figure(figsize=(8, batch * 4))
plt.subplot(batch, 2, 1)
plt.title("Code")
plt.subplot(batch, 2, 2)
plt.title("Generated Sample")

for i in range(batch):
    plt.subplot(batch, 2, i * 2 + 1)
    plt.imshow(priors[i], cmap='gray')
    plt.axis("off")

    plt.subplot(batch, 2, i * 2 + 2)
    plt.imshow(generated_samples[i].squeeze() + 0.5, cmap='gray')
    plt.axis("off")
plt.show()
