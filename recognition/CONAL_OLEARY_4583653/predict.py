from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

BATCH = 20


def VQVAEPredict(vqvae, slice_test):
    """
      vqvae: The instantiated VQVAE Model
      slice_test: The testing dataset
    """
    vqvae.plot(10, slice_test)


def PixelCNNPredict(vqvae, pixelCNN, slice_train, slice_test):
    """
      vqvae: The instantiated VQVAE Model
      pixelCNN: The instantiated pixelCNN Model
      slice_train: The training dataset
      slice_test: The test dataset
    """
    test = slice_test.take(1)
    for elem in test:
        elem = elem.numpy()
    testset = elem
    test = vqvae.encoder.predict(elem)
    out = pixelCNN.predict(test)
    out = np.expand_dims(out[0], axis=0)

    shape = ((BATCH,) + out.shape[1:])
    priors = tf.zeros(shape=shape)
    batch, rows, cols, embedding_count = priors.shape

    # Create the priors

    # Iterate over the priors pixel by pixel.
    for row in range(rows):
        for col in range(cols):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            x = pixelCNN(priors, training=False)
            dist = tfp.distributions.Categorical(logits=x)
            sampled = dist.sample()
            sampled2 = tf.one_hot(sampled, 256)
            priors = sampled2
    print(f"Prior shape: {priors.shape}")

    embedding_dim = vqvae.vq_layer.embedding_dim
    pretrained_embeddings = vqvae.vq_layer.embeddings
    pixels = tf.constant(priors, dtype="float32")

    quantized = tf.matmul(pixels, pretrained_embeddings, transpose_b=True)
    print(quantized.shape)
    print(sampled.shape)
    # Plot the priors codebooks
    for i in range(batch):
        plt.figure()
        plt.imshow(sampled[i])

    # Plot and save the generated brains
    generated_samples = vqvae.decoder.predict(quantized)
    for i in range(batch):
        plt.figure(figsize=(5, 6))
        plt.imshow(tf.squeeze(generated_samples[i]) * 255, cmap='gray')
        plt.title("Generated Sample")
        plt.axis("off")
        plt.show()
        print("SSIM: ", tf.reduce_mean(tf.image.ssim(
            generated_samples[i]*255, testset*255, max_val=255)))
        im = keras.utils.array_to_img(generated_samples[i])
        im.save(f"fig{i}.png")
