"""
train_pixel_cnn.py

Alex Nicholson (45316207)
11/10/2022

Contains the source code for training, validating, testing and saving your model. The model is imported from “modules.py” and the data loader is imported from “dataset.py”. Losses and metrics are plotted throughout training.

# TODO: Document this file
"""


import dataset
import modules
import utils
from tensorflow import keras
import tensorflow as tfs
import numpy as np
import os as os


if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    #                                HYPERPARAMETERS                               #
    # ---------------------------------------------------------------------------- #
    # EXAMPLES_TO_SHOW = 10
    
    NUM_EMBEDDINGS = 128
    NUM_RESIDUAL_BLOCKS = 2
    NUM_PIXELCNN_LAYERS = 2
    
    BATCH_SIZE = 128
    NUM_EPOCHS = 60
    VALIDATION_SPLIT = 0.1

    reuse_codebook_indices = True
    continue_training = False

    # ---------------------------------------------------------------------------- #
    #                                   LOAD DATA                                  #
    # ---------------------------------------------------------------------------- #
    # Import data loader from dataset.py
    (train_data, validate_data, test_data, data_variance) = dataset.load_dataset(max_images=3000, verbose=True)


    # ---------------------------------------------------------------------------- #
    #                          IMPORT TRAINED VQVAE MODEL                          #
    # ---------------------------------------------------------------------------- #
    # Import trained and saved model from file
    trained_vqvae_model = keras.models.load_model("./vqvae_saved_model")

    # ---------------------------------------------------------------------------- #
    #                     GENERATE TRAINING DATA FOR PIXEL CNN                     #
    # ---------------------------------------------------------------------------- #
    print("Generating pixelcnn training data...")
    # Generate the codebook indices.
    encoded_outputs = trained_vqvae_model.get_layer("encoder").predict(train_data)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])

    if reuse_codebook_indices == True and os.path.exists('./codebook_indices.csv'):
        print("Loading pre-computed codebook indices from file")
        # Pull the codebook indices from file
        codebook_indices = np.loadtxt('./codebook_indices.csv', delimiter=',')
    else:
        print("Calculating codebook indices")
        # Calculate the codebook indices from scratch
        codebook_indices = utils.get_code_indices_savedmodel(trained_vqvae_model.get_layer("vector_quantizer"), flat_enc_outputs)
        np.savetxt('./codebook_indices.csv', codebook_indices, delimiter=',')

    print("D")

    codebook_indices = codebook_indices.reshape(encoded_outputs.shape[:-1])
    print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")


    # ---------------------------------------------------------------------------- #
    #                                  BUILD MODEL                                 #
    # ---------------------------------------------------------------------------- #
    print("Building model...")
    
    if continue_training:
        # Continue the training of an aoldready part trained model
        pixel_cnn = keras.models.load_model("./pixelcnn_saved_model")
    else:
        # Start training from scratch
        pixelcnn_input_shape = trained_vqvae_model.get_layer("encoder").predict(train_data).shape[1:-1]
        pixel_cnn = modules.get_pixel_cnn(trained_vqvae_model, pixelcnn_input_shape, NUM_EMBEDDINGS, NUM_RESIDUAL_BLOCKS, NUM_PIXELCNN_LAYERS)

    
    pixel_cnn.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    pixel_cnn.summary()

    # ---------------------------------------------------------------------------- #
    #                                 RUN TRAINING                                 #
    # ---------------------------------------------------------------------------- #
    print("Training model...")
    pixel_cnn.fit(
        x=codebook_indices,
        y=codebook_indices,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_split=VALIDATION_SPLIT,
    )


    # ---------------------------------------------------------------------------- #
    #                                SAVE THE MODEL                                #
    # ---------------------------------------------------------------------------- #
    # Get the trained model
    trained_pixelcnn_model = pixel_cnn

    # Save the model to file as a tensorflow SavedModel
    trained_pixelcnn_model.save("./pixelcnn_saved_model")

    # ---------------------------------------------------------------------------- #
    #                                 FINAL RESULTS                                #
    # ---------------------------------------------------------------------------- #
    # Visualise the discrete codes
    examples_to_show = 10
    utils.visualise_codes(trained_vqvae_model, test_data, examples_to_show)

    # # Visualise novel generations from codes
    num_embeddings = 128
    utils.visualise_codebook_sampling(trained_vqvae_model, pixel_cnn, train_data, num_embeddings, examples_to_show)