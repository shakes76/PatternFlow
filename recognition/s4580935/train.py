from pickletools import optimize
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

import modules
import dataset

def main():
    #Get data file path locations
    train_images = dataset.FetchData('D:\\University\\2022 Sem 2\\COMP3710\\Project Report\\BranchFOlder\\keras_png_slices_data\\keras_png_slices_data\\keras_png_slices_train\\*')
    test_images = dataset.FetchData('D:\\University\\2022 Sem 2\\COMP3710\\Project Report\\BranchFOlder\\keras_png_slices_data\\keras_png_slices_data\\keras_png_slices_test\\*')
    validate_images = dataset.FetchData('D:\\University\\2022 Sem 2\\COMP3710\\Project Report\\BranchFOlder\\keras_png_slices_data\\keras_png_slices_data\\keras_png_slices_validate\\*')
    #Extract all the images in each file and do some pre-processing
    train = dataset.ImageExtract(train_images)
    test = dataset.ImageExtract(test_images)
    validate = dataset.ImageExtract(validate_images)
    #combine training and validation into one larger set for training
    Oasis = dataset.combine(train, validate)
    #Change test and validate set dimensions for later use
    test = np.squeeze(test)
    test = np.expand_dims(test, -1).astype("float32")
    validate = np.squeeze(validate)
    validate = np.expand_dims(validate, -1).astype("float32")
    #Check to make sure the train and validate sets have been combines
    #check that they have the right shape (256,256) and values between 0 and 1
    print(Oasis.shape)
    print(Oasis.min(), Oasis.max())
    #Check the summaries for the encoder, decoder and combined vqvae models
    modules.new_encoder(32).summary()
    modules.new_decoder(32).summary()
    modules.get_vqvae(32, 128).summary()
    #determine the var in the Oasis set
    variance = np.var(Oasis)
    #build, compile and fit model
    model = modules.VQVAE(variance, latent_dim=32, num_embeddings=128)
    model.compile(optimizer=keras.optimizers.Adam())
    history = model.fit(Oasis, epochs=30, batch_size=128)
    #Show Reconstruction Loss
    plt.subplot(211)
    plt.title('Reconstruction Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.plot(history.history['reconstruction_loss'])
