import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

















# checkpoint_path = "training/ckpt01.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# n_epochs = 200

# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     save_weights_only=True,
#     verbose=1,
#     save_freq=n_epochs*X_train.shape[0]
# )