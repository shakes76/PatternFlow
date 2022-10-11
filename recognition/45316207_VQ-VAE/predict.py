"""
predict.py

Alex Nicholson (45316207)
11/10/2022

Shows example usage of your trained model. Print out any results and / or provide visualisations where applicable

"""


import dataset
import utils
from tensorflow import keras
import numpy as np # TODO: Remove this later once we swap to the OASIS data


if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    #                                   LOAD DATA                                  #
    # ---------------------------------------------------------------------------- #
    # Import data loader from dataset.py
    # (train_data, test_data, validate_data) = dataset.load_dataset()

    # ------------------------------------ NEW ----------------------------------- #
    # Load and preprocess the MNIST dataset
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    x_train_scaled = (x_train / 255.0) - 0.5
    x_test_scaled = (x_test / 255.0) - 0.5

    data_variance = np.var(x_train / 255.0)
    # ------------------------------------ NEW ----------------------------------- #


    # ---------------------------------------------------------------------------- #
    #                             IMPORT TRAINED MODEL                             #
    # ---------------------------------------------------------------------------- #
    # Import trained and saved model from file
    trained_vqvae_model = keras.models.load_model("./vqvae_saved_model")


    # ---------------------------------------------------------------------------- #s
    #                                 FINAL RESULTS                                #
    # ---------------------------------------------------------------------------- #
    # Visualise the final results and calculate the structural similarity index (SSIM)
    idx = np.random.choice(len(x_test_scaled), 10)
    test_images = x_test_scaled[idx]
    reconstructions_test = trained_vqvae_model.predict(test_images)

    for test_image, reconstructed_image in zip(test_images, reconstructions_test):
        utils.show_subplot(test_image, reconstructed_image)