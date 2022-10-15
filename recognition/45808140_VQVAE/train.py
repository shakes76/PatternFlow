import numpy as np
from modules import *
from dataset import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

root_path = 'AD_NC'
img_shape = 256
no_epoch = 10
batch_size = 32

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

(train_data, test_data, train_var) = load_data(root_path, batch_size)

VQVAE = VQVAE_model(img_shape, train_var, 16, 128)
VQVAE.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4))
print(VQVAE.get_model().summary())

with tf.device('/GPU:0'):
    history = VQVAE.fit(train_data, epochs=no_epoch, batch_size=batch_size, callbacks=[ssim(test_data)])

VQVAE.save_weights('vq_weights')




