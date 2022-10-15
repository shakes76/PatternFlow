from dataset import *
from modules import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

(train_data, test_data, train_variance) = load_data('AD_NC')

def show_reconstructed(original, reconstructed, codebook_indices):
    plt.subplot(1, 3, 1)
    plt.imshow(original.numpy().squeeze())
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(codebook_indices)
    plt.title("Code")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed.squeeze())
    plt.title("Reconstructed")
    plt.axis("off")
    plt.show()

def VQVAE_result(vqvae, dataset):
    test_images = None
    for i in dataset.take(1):
        test_images = i

    encoder = vqvae.get_encoder()
    vq = vqvae.get_vq()
    recons = vqvae.predict(test_images)

    encoded_outputs = encoder.predict(test_images)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_ind = vq.get_code_indices(flat_enc_outputs)
    codebook_ind = codebook_ind.numpy().reshape(encoded_outputs.shape[:-1])

    for test_image, reconstructed_image, codebook in zip(test_images, recons, codebook_ind):
        show_reconstructed(test_image, reconstructed_image, codebook)

def get_codebooks(vq, embeds):
    def mapper(x):
        encoded_outputs = vq.get_encoder()(x)
        flat_enc_outputs = tf.reshape(encoded_outputs, [-1, tf.shape(encoded_outputs)[-1]])

        code_ind = vq.get_vq().get_code_indices(flat_enc_outputs)
        code_ind = tf.reshape(code_ind, tf.shape(encoded_outputs)[:-1])
        return code_ind
    
    return mapper

def generate_PixelCNN(vq, pcnn, n):
    encoded_outputs = vq.get_encoder().predict(train_data)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = get_codebooks(vq, vq.no_embeddings)

    codebook_data = train_data.map(codebook_indices)

    inputs = keras.Input(shape=pcnn.input_shape[1:])
    outputs = pcnn(inputs, training=False)
    dist = tfp.layers.DistributionLambda(tfp.distributions.Categorical)

    outputs = dist(outputs)
    sampler = keras.Model(inputs, outputs)
    sampler.trainable = False

    priors = np.zeros(shape=(n,) + (pcnn.input_shape)[1:])
    _, rows, cols = priors.shape

    for row in range(rows):
        for col in range(cols):
            probs = sampler.predict(priors)
            priors[:, row, col] = probs[:, row, col]

    embeddings = vq.get_vq().embeddings
    priors_one_hot = tf.one_hot(priors.astype('int32'), vq.no_embeddings).numpy()

    quant = tf.matmul(priors_one_hot.astype('float32'), embeddings, transpose_b=True)
    quant = tf.reshape(quant, (-1, *(encoded_outputs.shape[1:])))

    gen_samps = vq.get_decoder().predict(quant)

    for i in range(n):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i])
        plt.title("Generated codebook sample")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(gen_samps[i].squeeze(), cmap="gray")
        plt.title("Decoded sample")
        plt.axis("off")
        plt.show()
