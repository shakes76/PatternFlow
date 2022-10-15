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
import predict

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

    trained_vqvae_model = modules.model.vqvae
    idx = np.random.choice(len(test), len(test))
    test_images = test[idx]
    reconstructions_test = trained_vqvae_model.predict(test_images)
    simArray = []
    for test_image, recon_img in zip(test_images, reconstructions_test):
        simArray.append(predict.calculate_ssim(test_image, recon_img))
    #Determine the Average ssim over the dataset (test set)
    print(predict.average_ssim(simArray)) 
    trained_vqvae_model = model.vqvae
    idx = np.random.choice(len(test), 15)
    print(idx)
    test_images = test[idx]
    reconstructions_test = trained_vqvae_model.predict(test_images)

    #Show samples of the origional image along with the reconstructed image
    for test_image, reconstructed_image in zip(test_images, reconstructions_test):
        show_subplot(test_image, reconstructed_image)
        sim = calculate_ssim(test_image, reconstructed_image)
        print(sim)

    encoder = model.vqvae.get_layer("encoder")
    quantizer = model.vqvae.get_layer("vector_quantizer")

    encoded_outputs = encoder.predict(test_images)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

    for i in range(len(test_images)):
        plt.subplot(1, 2, 1)
        plt.imshow(test_images[i].squeeze() + 0.5)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(codebook_indices[i])
        plt.title("Code")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()