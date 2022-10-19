import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
import tensorflow_probability as tfp
from pixel import *


def show_subplot(original, reconstructed):
    SSIM = tf.image.ssim(original, reconstructed, max_val=1)
    plt.suptitle("SSIM: {}".format(SSIM))
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()

def reconstruct_oasis(vqvae_trainer, test_np):
    # Plot the reconstructions (randomly choose from the test set)
    trained_vqvae_model = vqvae_trainer.vqvae
    idx = np.random.choice(len(test_np), 10)
    test_images = test_np[idx]
    reconstructions_test = trained_vqvae_model.predict(test_images)

    for test_image, reconstructed_image in zip(test_images, reconstructions_test):
        show_subplot(test_image, reconstructed_image)

    # Plot the codes
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
        plt.show()

    return encoded_outputs.shape

# ====================================================
# def plot_codes(vqvae_trainer, test_images):
#     encoder = vqvae_trainer.vqvae.get_layer("encoder")
#     quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

#     encoded_outputs = encoder.predict(test_images)
#     flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
#     codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
#     codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

#     for i in range(len(test_images)):
#         plt.subplot(1, 2, 1)
#         plt.imshow(test_images[i].squeeze() + 0.5)
#         plt.title("Original")
#         plt.axis("off")

#         plt.subplot(1, 2, 2)
#         plt.imshow(codebook_indices[i])
#         plt.title("Code")
#         plt.axis("off")
#         plt.show()


# ==================================================
# This is test/predict using the Pixel model
# Create a mini sampler model.
def get_sampler(pixel_cnn_model):
    inputs = layers.Input(shape=pixel_cnn_model.input_shape[1:])
    outputs = pixel_cnn_model(inputs, training=False)
    categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
    outputs = categorical_layer(outputs)
    sampler = keras.Model(inputs, outputs)
    return sampler

# ============
# Create an empty array of priors.
def get_priors(pixel_cnn_model, sampler):
    batch = 10
    priors = np.zeros(shape=(batch,) + (pixel_cnn_model.input_shape)[1:])
    batch, rows, cols = priors.shape

    # Iterate over the priors because generation is done sequentially pixel by pixel.
    for row in range(rows):
        for col in range(cols):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            probs = sampler.predict(priors)
            # Use the probabilities to pick pixel values and append the values to the priors.
            priors[:, row, col] = probs[:, row, col]

    print(f"Prior shape: {priors.shape}")
    return priors


def get_priors2(vqvae_trainer, pixel_cnn, encoder_output_shape):
    n_priors = 10
    priors = tf.Variable(tf.zeros(shape=(n_priors,) + pixel_cnn.input_shape[1:], dtype=tf.int32))

    _, rows, cols = priors.shape

    for row in range(rows):
        for col in range(cols):
            print(f"\rrow: {row}, col: {col}", end="")
            dist = tfp.distributions.Categorical(logits=pixel_cnn(priors, training=False))
            probs = dist.sample()

            priors = priors[:, row, col].assign(probs[:, row, col])
    quantiser = vqvae_trainer.vqvae.get_layer("vector_quantizer")

    embeddings = quantiser.embeddings
    priors = tf.cast(priors, tf.int32)
    priors_one_hot = tf.one_hot(priors, vqvae_trainer.num_embeddings)
    priors_one_hot = tf.cast(priors_one_hot, tf.float32)
    quantised = tf.matmul(priors_one_hot, embeddings, transpose_b=True)
    quantised = tf.reshape(quantised, (-1, *(encoder_output_shape[1:])))

    # Generate novel images.
    # decoder = vqvae_trainer.vqvae.get_layer("decoder")
    # generated_samples = decoder.predict(quantised)
    return priors, quantised
# ================
# Perform an embedding lookup.
def quantize_priors(priors, vqvae_trainer, encoder_output_shape):
    quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")
    pretrained_embeddings = quantizer.embeddings
    priors_ohe = tf.one_hot(priors.astype("int32"), vqvae_trainer.num_embeddings).numpy()
    quantized = tf.matmul(
        priors_ohe, pretrained_embeddings, transpose_b=True
    )
    quantized = tf.reshape(quantized, (-1, *(encoder_output_shape[1:])))
    return quantized

# Generate novel images.
def generator(vqvae_trainer, quantized, priors):
    batch = 10
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


