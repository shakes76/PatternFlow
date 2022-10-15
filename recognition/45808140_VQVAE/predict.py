from dataset import *
from modules import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

(train_data, test_data, train_variance) = load_data('AD_NC')

def show_reconstructed(original, reconstructed, codebook_indices):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(original.numpy().squeeze())
    ax1.title.set_text("Original")
    ax1.axis("off")
    
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(codebook_indices)
    ax2.title.set_text("Code")
    ax2.axis("off")
    
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(reconstructed.squeeze())
    ax3.title.set_text("Reconstructed")
    ax3.axis("off")
    fig.show()
    fig.savefig('VQVAE_recons.png')

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

    inputs = keras.Input(shape=vq.get_encoder().output.shape[1:3])
    outputs = pcnn(inputs, training=False)
    dist = tfp.layers.DistributionLambda(tfp.distributions.Categorical)

    outputs = dist(outputs)
    sampler = keras.Model(inputs, outputs)
    sampler.trainable = False

    priors = np.zeros(shape=(n,) + (pcnn.input_shape)[1:])
    _, rows, cols = priors.shape

    for row in range(rows):
        for col in range(cols):
            probs = sampler.predict(priors, verbose=0)
            priors[:, row, col] = probs[:, row, col]

    embeddings = vq.get_vq().embeddings
    priors_one_hot = tf.one_hot(priors.astype('int32'), vq.no_embeddings).numpy()

    quant = tf.matmul(priors_one_hot.astype('float32'), embeddings, transpose_b=True)
    quant = tf.reshape(quant, (-1, *(encoded_outputs.shape[1:])))

    gen_samps = vq.get_decoder().predict(quant)

    for i in range(n):       
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(priors[i])
        ax1.title.set_text("Generated codebook sample")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(gen_samps[i].squeeze(), cmap="gray")
        ax2.title.set_text("Decoded sample")
        ax2.axis("off")
        fig.show()
        fig.savefig('generated.png')
