"""
predict.py

Alex Nicholson (45316207)
11/10/2022

Shows example usage of your trained model. Print out any results and / or provide visualisations where applicable

"""


import dataset
import utils
from tensorflow import keras
import numpy as np


if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    #                                   LOAD DATA                                  #
    # ---------------------------------------------------------------------------- #
    # Import data loader from dataset.py
    (train_data, validate_data, test_data, data_variance) = dataset.load_dataset(max_images=None, verbose=True)


    # ---------------------------------------------------------------------------- #
    #                             IMPORT TRAINED MODEL                             #
    # ---------------------------------------------------------------------------- #
    # Import trained and saved model from file
    trained_vqvae_model = keras.models.load_model("./vqvae_saved_model")


    # ---------------------------------------------------------------------------- #s
    #                                 FINAL RESULTS                                #
    # ---------------------------------------------------------------------------- #
    # Visualise the final results and calculate the structural similarity index (SSIM)
    examples_to_show = 10
    utils.show_reconstruction_examples(trained_vqvae_model, test_data, examples_to_show)