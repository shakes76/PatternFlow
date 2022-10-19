from dataset import *
from modules import *
from train import *
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

#Variables
root_path = 'AD_NC'
img_shape = 256
vq_epoch = 5
pcnn_epoch = 3
batch_size = 32
num_embeds = 64
result_path = 'results'

#Make results directory to store results and save models
if not os.path.isdir(result_path):
    os.mkdir(result_path)

def show_reconstructed(original, reconstructed, codebook_indices, n, ssim):
    """Function to plot original, codebook and decoded images"""
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
    fig.text(.5, .3, 'ssim = {}'.format(ssim), ha='center')
    fig.savefig('{}/VQVAE_recons_{}.png'.format(result_path, n))
    fig.show()

def VQVAE_result(vqvae, dataset):
    """Function to plot the results of VQVAE model"""
    test_images = None
    #Take an image from the test dataset
    for i in dataset.take(1):
        test_images = i

    #Get components of VQVAE
    encoder = vqvae.get_encoder()
    vq = vqvae.get_vq()

    #Pass images into models to create codebook and decoded images from input
    recons = vqvae.predict(test_images)
    encoded_outputs = encoder.predict(test_images)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_ind = vq.get_code_indices(flat_enc_outputs)
    codebook_ind = codebook_ind.numpy().reshape(encoded_outputs.shape[:-1])

    count = 0
    #Plot images
    for test_image, reconstructed_image, codebook in zip(test_images, recons, codebook_ind):
        total_ssim = 0.0
        total_ssim += tf.math.reduce_sum(tf.image.ssim(test_image, reconstructed_image, max_val=1.0))
        show_reconstructed(test_image, reconstructed_image, codebook, count, total_ssim.numpy())
        count += 1

        if count == 5:
            break

def generate_PixelCNN(vq, pcnn, n):
    """Function to generate new brain images from distribution"""
    encoded_outputs = vq.get_encoder().predict(train_data)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = get_codebooks(vq, vq.no_embeddings)

    #map codebook to all the training data
    codebook_data = train_data.map(codebook_indices)

    #Get PixelCNN model to generate samples from distribution
    inputs = keras.Input(shape=vq.get_encoder().output.shape[1:3])
    outputs = pcnn(inputs, training=False)
    dist = tfp.layers.DistributionLambda(tfp.distributions.Categorical)

    outputs = dist(outputs)
    sampler = keras.Model(inputs, outputs)
    sampler.trainable = False

    #Initialise priors 
    priors = np.zeros(shape=(n,) + (pcnn.input_shape)[1:])
    _, rows, cols = priors.shape

    #Update priors pixel by pixel
    for row in range(rows):
        for col in range(cols):
            probs = sampler.predict(priors, verbose=0)
            priors[:, row, col] = probs[:, row, col]

    embeddings = vq.get_vq().embeddings
    priors_one_hot = tf.one_hot(priors.astype('int32'), vq.no_embeddings).numpy()

    quant = tf.matmul(priors_one_hot.astype('float32'), embeddings, transpose_b=True)
    quant = tf.reshape(quant, (-1, *(encoded_outputs.shape[1:])))

    #Generate samples
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
        fig.savefig('{}/generated_{}.png'.format(result_path, i))
        fig.show()


(train_data, test_data, train_var) = load_data(root_path, batch_size)

vqvae_trained = vq_train(train_data=train_data, test_data=test_data, train_var=train_var, 
                         vq_trained = None, img_shape=img_shape, latent_dim=32, embed_num=32, 
                         result_path=result_path, vq_epoch=vq_epoch)

VQVAE_result(vqvae_trained, test_data)

pcnn_trained = pcnn_train(vqvae_trained, train_data, result_path, pcnn_trained=None, 
                          pcnn_epoch=pcnn_epoch)

generate_PixelCNN(vqvae_trained, pcnn_trained, n=10)
