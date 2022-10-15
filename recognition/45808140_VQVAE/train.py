import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
from modules import *

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