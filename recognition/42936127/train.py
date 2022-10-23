import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from PIL import Image
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
seed = 123
batch_size = 64
img_height = 256
img_width = 256
image_shape = (img_height, img_width, 3)

from modules import *
from tools import *

from dataset import train_ds, val_ds, train_variance

load_model = True
print("VQVAE Training")
print(" ")
print(" ")

# user_latent_dim = int(input("Enter latent dimensionality : "))
# user_num_embeddings = int(input("Enter number of embeddings : "))
# user_epochs = int(input("Enter training epochs : "))
# user_batch_size = int(input("Enter batch size : "))

# print("Training with ")
# print("Latent dimensionality : ", user_latent_dim)
# print("Embeddings : ", user_num_embeddings)
# print("Epochs : ", user_epochs)
# print("Batch size : ", user_batch_size)

vqvae_trainer = VQVAETrainer(train_variance, 
    latent_dim = 16, 
    num_embeddings = 64, 
    image_shape = image_shape
)

vqvae_trainer.compile(optimizer=keras.optimizers.Adam())

history = vqvae_trainer.fit(
    x = train_ds,
    validation_data = val_ds,
    epochs = 10,
    batch_size = 64,
    use_multiprocessing = True,
    verbose = 1
)
vqvae_trainer.vqvae.save("saved_models")
vqvae = vqvae_trainer.vqvae
print("Trained and saved model")

show_history(history.history)


test_images = val_ds.take(1)
reconstructed_test_images = vqvae.predict(test_images)

len(reconstructed_test_images)

for test_image, reconstructed_image in zip(test_images, reconstructed_test_images):
    show_original_vs_reconstructed(test_image[1],reconstructed_test_images[1])

encoder = vqvae_trainer.vqvae.get_layer("encoder")
quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

encoded_outputs = encoder.predict(test_images)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

for test_image, reconstructed_image in zip(test_images, reconstructed_test_images):
    test_image = test_image[10]
    reconstructed_image = reconstructed_image[10]

for i in range(len(test_images)):
    plt.subplot(1, 2, 1)
    plt.imshow(test_image / 255.0)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(codebook_indices[i])
    plt.title("Code")
    plt.axis("off")
    plt.show()

# PixelCNN hyperparameters
num_residual_blocks = 1
num_pixelcnn_layers = 1
pixelcnn_input_shape = encoded_outputs.shape[1:-1]
print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")

pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
ohe = tf.one_hot(pixelcnn_inputs, vqvae_trainer.num_embeddings)
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
    filters=vqvae_trainer.num_embeddings, kernel_size=1, strides=1, padding="valid"
)(x)

pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")

encoded_outputs = encoder.predict(train_ds.take(5))
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)

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
pixel_cnn.save("saved_models/pixelcnn/")

inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
outputs = pixel_cnn(inputs, training=False)
categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
outputs = categorical_layer(outputs)
sampler = keras.Model(inputs, outputs)

batch = 10
priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
batch, rows, cols = priors.shape

# Iterate over the priors because generation has to be done sequentially pixel by pixel.
for row in range(rows):
    for col in range(cols):
        # Feed the whole array and retrieving the pixel value probabilities for the next
        # pixel.
        probs = sampler.predict(priors, verbose = 0)
        # Use the probabilities to pick pixel values and append the values to the priors.
        priors[:, row, col] = probs[:, row, col]

#print(f"Prior shape: {priors.shape}")

pretrained_embeddings = quantizer.embeddings
priors_ohe = tf.one_hot(priors.astype("int32"), vqvae_trainer.num_embeddings).numpy()
quantized = tf.matmul(
    priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
)
quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))


decoder = vqvae_trainer.vqvae.get_layer("decoder")
generated_samples = decoder.predict(quantized)

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
