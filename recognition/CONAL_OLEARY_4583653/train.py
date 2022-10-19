import modules
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

BATCH = 20


def trainVQVAE(vqvae, slice_train, slice_test):
    vqvae.fit(slice_train, epochs=10)
    vqvae.plot(10, slice_test)


def trainPixelCNN(vqvae, pixelCNN, slice_train, slice_test):
    encoded_outputs = vqvae.encoder.predict(slice_test)
    shape = tf.shape(encoded_outputs).numpy()

    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])

    codebook_indices = vqvae.vq_layer.get_code_indices(flat_enc_outputs)

    codebook_indices = codebook_indices.numpy().reshape(
        encoded_outputs.shape[:-1])
    codebook_indices = tf.one_hot(codebook_indices, 256)

    print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")

    pixelCNN.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                     loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    print("Beginning training of PixelCNN\n")
    pixelCNN.fit(x=codebook_indices, y=codebook_indices,
                 validation_split=0.25, epochs=200)

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
    for i in range(batch):
        plt.figure()
        plt.imshow(sampled[i])

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


def train(slice_train, slice_test):
    vqvae = modules.VQVAE()
    vqvae.compile(optimizer=keras.optimizers.Adam())
    print("Beginning training of VQVAE\n")
    trainVQVAE(vqvae, slice_train, slice_test)
    pixelCNN = modules.PixelCNN(25, 10, 256)
    trainPixelCNN(vqvae, pixelCNN, slice_train, slice_test)
