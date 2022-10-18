import numpy as np
import matplotlib.pyplot as plt
from modules import *
from dataset import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

#Base variables
root_path = 'AD_NC'
img_shape = 256
num_embeds = 128
latent_dim = 32

vq_epoch = 20
pcnn_epoch = 10
batch_size = 32
no_resid = 2
no_pcnn_layers = 2

"""Custom metric to calculate average similarity of reconstructions"""
class ssim(keras.callbacks.Callback):
    def __init__(self, validation):
        super(ssim, self).__init__()
        self.val = validation
        
    def on_epoch_end(self, epoch, logs):
        """Calculate average similarity of reconstructions at the end of epoch"""
        total_count = 0.0
        total_ssim = 0.0
        
        # reconstruct images from batched samples to calculate ssim
        for batch in self.val:
            recon = self.model.predict(batch, verbose=0)
            total_ssim += tf.math.reduce_sum(tf.image.ssim(batch, recon, max_val=1.0))
            total_count += batch.shape[0]
            
        logs['avg_ssim'] = (total_ssim / total_count).numpy()


def get_codebooks(vq, embeds):
    """Custom function to batch codebooks"""
    def mapper(x):
        encoded_outputs = vq.get_encoder()(x)
        flat_enc_outputs = tf.reshape(encoded_outputs, [-1, tf.shape(encoded_outputs)[-1]])

        code_ind = vq.get_vq().get_code_indices(flat_enc_outputs)
        code_ind = tf.reshape(code_ind, tf.shape(encoded_outputs)[:-1])
        return code_ind
    
    return mapper

def vq_train(train_data, test_data, train_var, result_path, vq_trained=None, img_shape=img_shape, 
    latent_dim=latent_dim, embed_num=num_embeds,vq_epoch=vq_epoch):
    """Function to fit/train VQVAE model"""

    #Use pre-trained vqvae model if specified
    if vq_trained is None:
        VQVAE = VQVAE_model(img_shape, train_var, latent_dim=latent_dim, 
                            no_embeddings=embed_num)
    else:
        VQVAE = vqvae_trained

    VQVAE.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4))
    print(VQVAE.get_model().summary())

    #Fitting VQVAE
    with tf.device('/GPU:0'):
        history = VQVAE.fit(train_data, epochs=vq_epoch, batch_size=batch_size, callbacks=[ssim(test_data)])

    #Plot of loss performance
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['vq_loss'], label = 'vq_loss')
    plt.plot(history.history['reconstruction_loss'], label = 'reconstruction_loss')
    plt.plot(history.history['avg_ssim'], label = 'similarity')
    plt.legend(loc='upper right')
    plt.ylim([0,1])
    plt.xlabel('Epoch')
    plt.savefig('{}/vq_result_graph.png'.format(result_path))

    #save weights to reuse
    VQVAE.save_weights('{}/vq_weights'.format(result_path))
    return VQVAE

def pcnn_train(VQVAE, train_data, result_path, pcnn_trained=None, pcnn_epoch=pcnn_epoch):
    codebook_mapper = get_codebooks(VQVAE, VQVAE.no_embeddings)
    codebook_data = train_data.map(codebook_mapper)
    """Function to fit/train PixelCNN model"""

    #Use pre-trained model if specified
    if pcnn_trained is None:
        pcnn = PixelCNN(VQVAE.encoder.output.shape[1:-1], VQVAE)
    else:
        pcnn = pcnn_trained

    print(pcnn.get_model().summary())
    pcnn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    #Fitting model
    with tf.device('/GPU:0'):
        history = pcnn.fit(codebook_data, batch_size=batch_size, epochs=pcnn_epoch)

    #Plot and save loss performance
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.savefig('{}/pcnn_result_graph.png'.format(result_path))

    #save weights to reuse
    pcnn.save_weights('{}/pcnn_weights'.format(result_path))

    return pcnn


    





