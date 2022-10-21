import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from PIL import Image
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
seed = 123
batch_size = 64
img_height = 256
img_width = 256
image_shape = (img_height, img_width, 3)

from modules import *
from tools import *

load_model = True

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "keras_png_slices_data/slices/", 
    labels = None,
    validation_split = 0.3,
    subset = "both",
    seed = seed,
    image_size = (img_height, img_width)
)

# train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
#     "keras_png_slices_data/slices/", 
#     labels = None,
#     validation_split = 0.3,
#     subset = "both",
#     seed = seed,
#     image_size = (img_height, img_width)
# )


# if(load_model):
#     vqvae_trainer = VQVAETrainer(0.05, latent_dim=16, num_embeddings=128, image_shape = image_shape)
#     vqvae_trainer.vqvae = keras.models.load_model("saved_models")
#     vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
#     vqvae = vqvae_trainer.vqvae
#     print("loaded model")
# else:
#     #data_variance = tf.math.reduce_variance(train_ds)
#     vqvae_trainer = VQVAETrainer(0.05, latent_dim=16, num_embeddings=128, image_shape = image_shape)
#     vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
#     vqvae_trainer.fit(
#         x = train_ds,
#         validation_data = val_ds,
#         epochs = 1,
#         use_multiprocessing = True,
#         verbose = 1
#     )
#     vqvae_trainer.vqvae.save("saved_models")
#     vqvae = vqvae_trainer.vqvae
#     print("trained and saved model")