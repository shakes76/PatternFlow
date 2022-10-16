import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf
import zipfile
import os
import dataset, modules

figure_path = "figures/"

# Load data
train, test, validate = dataset.load_data()

# Train VQVAE
vqvae_trainer = modules.VQVAETrainer(latent_dim=256, num_embeddings=256)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
vqvae_trainer.fit(train, epochs=100, batch_size=8, steps_per_epoch=len(train)/8)

# Plot learning
plt.plot(vqvae_trainer.history.history['reconstruction_loss'], label='reconstruction_loss')
plt.plot(vqvae_trainer.history.history['vqvae_loss'], label = 'vqvae_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.savefig(os.path.join(figure_path, "training_plot"))

# Reconstructions
trained_vqvae_model = vqvae_trainer.vqvae
idx = np.random.choice(len(test), 10)
test_images = test[idx]
reconstructions_test = trained_vqvae_model.predict(test_images)

i = 0
for test_image, reconstructed_image in zip(test_images, reconstructions_test):
    filename = os.path.join(figure_path, "reconstruction_" + str(i) + ".png")
    i += 1
    modules.save_subplot(test_image, reconstructed_image, filename)

# Encoding
encoder = vqvae_trainer.vqvae.get_layer("encoder")
quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

encoded_outputs = encoder.predict(test_images)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

for i in range(len(test_images)):
    plt.subplot(1, 2, 1)
    plt.imshow(test_images[i].squeeze() + 0.5)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(codebook_indices[i])
    plt.title("Code")
    plt.axis("off")
    filename = os.path.join(figure_path, "codebook_" + str(i) + ".png")
    plt.savefig(filename)

# PixelCNN
num_residual_blocks = 7
num_pixelcnn_layers = 2
pixelcnn_input_shape = encoded_outputs.shape[1:-1]

pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
ohe = tf.one_hot(pixelcnn_inputs, vqvae_trainer.num_embeddings)
x = modules.PixelConvLayer(
    mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
)(ohe)

for _ in range(num_residual_blocks):
    x = modules.ResidualBlock(filters=128)(x)

for _ in range(num_pixelcnn_layers):
    x = modules.PixelConvLayer(
        mask_type="B",
        filters=128,
        kernel_size=1,
        strides=1,
        activation="relu",
        padding="valid",
    )(x)
    x = layers.BatchNormalization()(x)


out = keras.layers.Conv2D(
    filters=256, kernel_size=1, strides=1, padding="valid"
)(x)
pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")

encoded_outputs = encoder.predict(train)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

pixel_cnn.compile(
    #optimizer=keras.optimizers.Adam(3e-4),
    optimizer=keras.optimizers.Adam(3e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
pixel_cnn.fit(
    x=codebook_indices,
    y=codebook_indices,
    batch_size=32,
    epochs=1000,

    validation_split=0.25,
)

# Create a mini sampler model.
inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
outputs = pixel_cnn(inputs, training=False)
categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
outputs = categorical_layer(outputs)
sampler = keras.Model(inputs, outputs)

# Create an empty array of priors.
batch = 100
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

# Plot Decoded samples
for i in range(batch):
    plt.subplot(1, 2, 1)
    plt.imshow(priors[i])
    plt.title("Code")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(generated_samples[i].squeeze() + 0.5)
    plt.title("Generated Sample")
    plt.axis("off")
    filename = os.path.join(figure_path, "decoded_" + str(i) + ".png")
    plt.savefig(filename)

# Plot learning
plt.figure()
plt.plot(pixel_cnn.history.history['loss'], label='loss')
plt.plot(pixel_cnn.history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.savefig(os.path.join(figure_path, "cnn_plot"))

# Check SSIM
for i in range(len(generated_samples)):
    img1 = train[i]
    img2 = generated_samples[i]
    print(tf.image.ssim(img1, img2, 1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03))