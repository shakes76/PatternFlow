import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(4)
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from modules import VQVAE, VectorQuantiser, PixelCNN
from constants import batch_size, variance, latent_dimensions, num_embeddings, n_residual_blocks, n_pixel_cnn_layers, n_epochs, vqvae_epochs
from dataset import x_train, x_test
from util import *



# Bug fix - my computer has some issue that is fixed using this one-liner
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print(f"========================================================================")
print(f"                             Load Dataset                               ")
print(f"========================================================================")
print(f"[ OK ] Load Dataset")
print(f"The training set contains {len(x_train) * batch_size} samples")
print(f"The testing set contains {len(x_test) * batch_size} samples")

# Train the VQVAE model.
print(f"========================================================================")
print(f"                             Train VQVAE                                ")
print(f"========================================================================")
vqvae = VQVAE(variance, latent_dimensions, num_embeddings)
vqvae.compile(optimizer=keras.optimizers.Adam())
vqvae.fit(x_train, epochs=vqvae_epochs, batch_size=batch_size)

# Evaluate
test_images, reconstructed = compare_reconstructions(vqvae, x_test, 10)
# tf.keras.models.save_model(model=vqvae, filepath="./model_weights/vqvae")
vqvae.save("vqvae", "./model_weights/vqvae", save_format="tf")

# Train PixelCNN
print(f"========================================================================")
print(f"                            Train PixelCNN                              ")
print(f"========================================================================")
encoder = vqvae.get_layer("encoder")
quantiser = vqvae.get_layer("quantiser")

outputs = encoder.predict(x_train)
flattened = outputs.reshape(-1, outputs.shape[-1])

code_indices = quantiser.get_code_indices(flattened)
code_indices = tf.reshape(code_indices, outputs.shape[:-1])

pixel_cnn = PixelCNN(vqvae.get_layer("encoder").out_shape, num_embeddings)
pixel_cnn.compile(
    optimizer=Adam(learning_rate=(0.0003)),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
print(pixel_cnn.summary())
pixel_cnn.fit(x=code_indices, y=code_indices, batch_size=64, epochs=n_epochs, validation_split=0.1)
pixel_cnn.save("./model_weights/pixel_cnn")

# tf.keras.models.save(model=pixel_cnn, filepath="./model_weights/pixel_cnn")
