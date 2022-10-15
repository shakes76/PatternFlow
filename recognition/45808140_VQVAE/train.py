import numpy as np
import matplotlib.pyplot as plt
from modules import *
from dataset import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

root_path = 'AD_NC'
img_shape = 256
vq_epoch = 5
pcnn_epoch = 3
batch_size = 32

no_resid = 2
no_pcnn_layers = 2

class ssim(keras.callbacks.Callback):
    def __init__(self, validation):
        super(ssim, self).__init__()
        self.val = validation
        
    def on_epoch_end(self, epoch, logs):
        total_count = 0.0
        total_ssim = 0.0
        
        for batch in self.val:
            recon = self.model.predict(batch, verbose=0)
            total_ssim += tf.math.reduce_sum(tf.image.ssim(batch, recon, max_val=1.0))
            total_count += batch.shape[0]
            
        logs['avg_ssim'] = (total_ssim / total_count).numpy()

def get_codebooks(vq, embeds):
    def mapper(x):
        encoded_outputs = vq.get_encoder()(x)
        flat_enc_outputs = tf.reshape(encoded_outputs, [-1, tf.shape(encoded_outputs)[-1]])

        code_ind = vq.get_vq().get_code_indices(flat_enc_outputs)
        code_ind = tf.reshape(code_ind, tf.shape(encoded_outputs)[:-1])
        return code_ind
    
    return mapper

def vq_train(train_data, test_data, train_var, img_shape):
    VQVAE = VQVAE_model(img_shape, train_var, 16, 128)
    VQVAE.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4))
    print(VQVAE.get_model().summary())

    with tf.device('/GPU:0'):
        history = VQVAE.fit(train_data, epochs=vq_epoch, batch_size=batch_size, callbacks=[ssim(test_data)])

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['vq_loss'], label = 'vq_loss')
    plt.plot(history.history['reconstruction_loss'], label = 'reconstruction_loss')
    plt.plot(history.history['avg_ssim'], label = 'similarity')
    plt.legend(loc='upper right')
    plt.ylim([0,5])
    plt.xlabel('Epoch')
    plt.show()
    plt.savefig('vq_result_graph.png')

    VQVAE.save_weights('vq_weights')
    return VQVAE

def pcnn_train(VQVAE, train_data):
    codebook_mapper = get_codebooks(VQVAE, VQVAE.no_embeddings)
    codebook_data = train_data.map(codebook_mapper)

    pcnn = PixelCNN(VQVAE.encoder.output.shape[1:-1], VQVAE)
    print(pcnn.get_model().summary())
    pcnn.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    with tf.device('/GPU:0'):
        history = pcnn.fit(codebook_data, batch_size=batch_size, epochs=pcnn_epoch)

    plt.plot(history.history['loss'], label='loss')
    plt.show()
    plt.savefig('pcnn_result_graph.png')

    pcnn.save_weights('pcnn_weights')

    return PCNN


    





