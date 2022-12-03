"""
predict.py

Alex Nicholson (45316207)
11/10/2022

Shows example usage of the trained model with visualisations of it's output results

"""


import dataset
import utils
import modules
from tensorflow import keras


if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    #                                   LOAD DATA                                  #
    # ---------------------------------------------------------------------------- #
    # Import data loader from dataset.py
    (train_data, validate_data, test_data, data_variance) = dataset.load_dataset(max_images=None, verbose=True)


    # ---------------------------------------------------------------------------- #
    #                          IMPORT TRAINED VQVAE MODEL                          #
    # ---------------------------------------------------------------------------- #
    # Import trained and saved vqvae model from file
    trained_vqvae_model = keras.models.load_model("./vqvae_saved_model", custom_objects={'VectorQuantizer': modules.CustomVectorQuantizer})


    # ---------------------------------------------------------------------------- #
    #                         IMPORT TRAINED PIXELCNN MODEL                        #
    # ---------------------------------------------------------------------------- #
    # Import trained and saved pixelcnn model from file
    trained_pixelcnn_model = keras.models.load_model("./pixelcnn_saved_model")


    # ---------------------------------------------------------------------------- #
    #                                 FINAL RESULTS                                #
    # ---------------------------------------------------------------------------- #


    examples_to_show = 10

    # # Visualise the final results and calculate the structural similarity index (SSIM)
    utils.show_reconstruction_examples(trained_vqvae_model, test_data, examples_to_show)

    # # Visualise the discrete codes
    utils.visualise_codes(trained_vqvae_model, test_data, examples_to_show)

    # # Visualise novel generations from codes
    num_embeddings = 128
    utils.visualise_codebook_sampling(trained_vqvae_model, trained_pixelcnn_model, train_data, num_embeddings, examples_to_show)