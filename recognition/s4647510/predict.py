import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf
import os
import dataset, modules

figure_path = "figures/"
num_embeddings = 256

train, test = dataset.load_data()

# Based off Keras VQVAE tutorial: https://keras.io/examples/generative/vq_vae/
# and Keras PixelCNN tutorial: https://keras.io/examples/generative/pixelcnn/

# Load in the trained models
vqvae = keras.models.load_model('models/vqvae_model')
pixel_cnn = keras.models.load_model('models/pixel_cnn_model')


idx = np.random.choice(len(test), 10)
test_images = test[idx]
encoder = vqvae.get_layer("encoder")
encoded_outputs = encoder.predict(test_images)
quantizer = vqvae.get_layer("vector_quantizer")

# Create a mini sampler model.
inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
outputs = pixel_cnn(inputs, training=False)
categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
outputs = categorical_layer(outputs)
sampler = keras.Model(inputs, outputs)

# Create an empty array of priors.
batch = 10
priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
batch, rows, cols = priors.shape

# Iterate over the priors because generation has to be done sequentially pixel by pixel.
for row in range(rows):
    for col in range(cols):
        probs = sampler.predict(priors)
        priors[:, row, col] = probs[:, row, col]

print(f"Prior shape: {priors.shape}")

# Perform an embedding lookup.
pretrained_embeddings = quantizer.embeddings
priors_ohe = tf.one_hot(priors.astype("int32"), num_embeddings).numpy()
quantized = tf.matmul(
    priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
)
quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))

# Generate novel images.
decoder = vqvae.get_layer("decoder")
generated_samples = decoder.predict(quantized)

# Plot Decoded samples
for i in range(batch):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(priors[i])
    plt.title("Code")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(generated_samples[i].squeeze() + 0.5)
    plt.title("Generated Sample")
    plt.axis("off")

    img1 = test_images[i]
    img2 = generated_samples[i]
    plt.figtext(0.5,0.2, "ssim:" + str(tf.image.ssim(img1, img2, 1))[10:17], size=12, ha="center")
    filename = os.path.join(figure_path, "decoded_" + str(i) + ".png")
    plt.savefig(filename)
    plt.close()