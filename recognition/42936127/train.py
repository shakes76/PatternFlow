import numpy as np
import matplotlib.pyplot as plpt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from modules import *
from dataset import train_ds, val_ds

#assert isinstance(ds_train.keys(), dict) #

#tfds.show_examples(ds_train, builder.info)
image_size = 256

input_shape = (None, image_size, image_size, 3)


train_ds = np.asarray(list(train_ds.unbatch()))

data_variance = tf.math.reduce_variance(train_ds)
print("HEYYYYYYY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(data_variance)
vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128, input_shape=input_shape)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())

#np.asarray(list(dataset.unbatch()))

# vqvae_trainer.fit(
#     train_ds.unbatch(),
#     validation_data = val_ds,
#     epochs = 5
# )


print("all done")


vqvae_trainer.vqvae.save_model("saved_models")