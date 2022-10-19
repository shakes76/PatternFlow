import modules
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import predict


def trainVQVAE(vqvae, slice_train, slice_test):
    """
      vqvae: The instantiated VQVAE Model
      slice_train: The training dataset
      slice_test: The test dataset
    """
    vqvae.fit(slice_train, epochs=10)


def trainPixelCNN(vqvae, pixelCNN, slice_train, slice_test):
    """
      vqvae: The instantiated VQVAE Model
      pixelCNN: The instantiated pixelCNN Model
      slice_train: The training dataset
      slice_test: The test dataset
    """
    encoded_outputs = vqvae.encoder.predict(slice_test)
    shape = tf.shape(encoded_outputs).numpy()

    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])

    codebook_indices = vqvae.vq_layer.get_code_indices(flat_enc_outputs)

    codebook_indices = codebook_indices.numpy().reshape(
        encoded_outputs.shape[:-1])
    codebook_indices = tf.one_hot(codebook_indices, 256)

    print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")

    pixelCNN.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                     loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    print("Beginning training of PixelCNN\n")
    pixelCNN.fit(x=codebook_indices, y=codebook_indices,
                 validation_split=0.25, epochs=200)
    predict.PixelCNNPredict(vqvae, pixelCNN, slice_train, slice_test)


def train(slice_train, slice_test):
    """
      slice_train: The training dataset
      slice_test: The test dataset
    """
    vqvae = modules.VQVAE()
    vqvae.compile(optimizer=keras.optimizers.Adam())
    print("Beginning training of VQVAE\n")
    trainVQVAE(vqvae, slice_train, slice_test)
    predict.VQVAEPredict(vqvae, slice_test)
    pixelCNN = modules.PixelCNN(25, 10, 256)
    trainPixelCNN(vqvae, pixelCNN, slice_train, slice_test)
