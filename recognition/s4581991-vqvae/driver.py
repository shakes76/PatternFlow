import os
from typing import Tuple

import keras.models
import keras.preprocessing.image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
import tensorflow_probability as tfp

import model

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# Constants and Parameters

OASIS_TRAIN_DIR = os.path.join(os.curdir, "keras_png_slices_data", "keras_png_slices_train")
OASIS_TEST_DIR = os.path.join(os.curdir, "keras_png_slices_data", "keras_png_slices_test")

INPUT_IMAGE_SHAPE = (96, 96, 3)

NUM_LATENT_DIMS = 16
NUM_EMBEDDINGS = 128

NUM_RECREATION_TEST_IMAGES = 20
NUM_SAMPLE_IMAGES_GENERATED = 20
GENERATED_IMAGE_DIR = os.path.join(os.curdir, "generated")

VQVAE_EPOCHS = 50
VQVAE_BATCH_SIZE = 128
VQVAE_VALIDATION_SPLIT = 0.1

PIXELCNN_EPOCHS = 20
PIXELCNN_BATCH_SIZE = 128
PIXELCNN_VALIDATION_SPLIT = 0.1

# Functions for OASIS

def load_oasis_data(path):
    '''
    Loads preprocessed OASIS Brain data
    '''
    files = os.listdir(path)
    oasis_images = []
    for filename in files:
        file_path = os.path.join(path, filename)
        image = keras.preprocessing.image.load_img(file_path, 
                target_size=INPUT_IMAGE_SHAPE[:2])
        oasis_images.append(
                keras.preprocessing.image.img_to_array(image) / 255)

    return np.array(oasis_images)

def run_oasis():
    '''
    Trains and tests the VQVAE model, then trains the PixelCNN model and uses
    the decoder from the VQVAE model to generate images of brains. The dataset
    used is the OASIS dataset.
    '''
    print("Loading data...")
    print(OASIS_TRAIN_DIR)
    train_data = load_oasis_data(OASIS_TRAIN_DIR)
    print(f"Loaded {train_data.shape[0]} training images")
    test_data = load_oasis_data(OASIS_TEST_DIR)
    print(f"Loaded {test_data.shape[0]} testing images")
    print("Loading complete!")
    
    print("Start VQVAE training...")
    history, vqvae = train_vqvae(train_data, INPUT_IMAGE_SHAPE,
        NUM_LATENT_DIMS, NUM_EMBEDDINGS)
    print("Training complete.")

    plot_oasis_vqvae_loss(history.history["loss"], history.history["val_loss"])

    recreated, recreated_ssim = test_vqvae(vqvae, test_data[:NUM_RECREATION_TEST_IMAGES])
    print(f"The SSIM is {recreated_ssim}")
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(recreated[i].squeeze())
        plt.axis("off")
    plt.title("'Recreated' Brain MRIs using VQ-VAE")
    plt.savefig("recreation.png")
    print("See recreation.png for recreated test images")

    history = generate_and_plot_with_pixel_cnn(vqvae, train_data)
    plot_oasis_pixelcnn_loss(history.history["loss"], history.history["val_loss"])

    print("Done.")

def plot_oasis_vqvae_loss(training_loss, validation_loss):
    '''
    Plot the loss achieved while training the VQVAE model
    '''
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.title("OASIS VQ-VAE Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.savefig("VQVAE_loss.png")

def plot_oasis_pixelcnn_loss(training_loss, validation_loss):
    '''
    Plot the loss achieved while training the PixelCNN model
    '''
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.title("OASIS PixelCNN Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.savefig("PixelCNN_loss.png")


# VQ VAE Model functions

def generate_and_plot_with_pixel_cnn(vqvae_model: keras.models.Model, train_data):
    '''
    Train PixelCNN, generate images and save them to disk.

    Parameters:
        - vqvae_model: a trained VQVAE model with layers named "encoder",
            "decoder" and "vector_quantizer".
        - train_data: Data to use to train the PixelCNN
    '''
    encoder: keras.layers.Layer = vqvae_model.get_layer("encoder")
    decoder: keras.layers.Layer = vqvae_model.get_layer("decoder")
    vector_quantizer = vqvae_model.get_layer("vector_quantizer")

    encoded = encoder.predict(train_data)
    flattened_encoded = encoded.reshape(-1, encoded.shape[-1])
    quantized = vector_quantizer.encode(flattened_encoded)
    quantized = quantized.numpy().reshape(encoded.shape[:-1])

    latent_height = quantized.shape[1]
    latent_width = quantized.shape[2]
    history, pixel_cnn = train_pixel_cnn(quantized, quantized,
            latent_width, latent_height,
            vector_quantizer.number_of_embeddings)
    generated_samples = generate_using_pixel_cnn(
            pixel_cnn, decoder, vector_quantizer, 
            NUM_SAMPLE_IMAGES_GENERATED, encoded.shape[1:])
    save_generated_samples(generated_samples, NUM_SAMPLE_IMAGES_GENERATED)

    return history

def train_vqvae(train_data: np.array, sample_shape: Tuple[int, int, int],
        latent_dimensions: int, number_of_embeddings: int):
    '''
    Train the VQ VAE model and return relavent stats and the trained model.

    Parameters:
        - train_data: the training data with shape (N, *(sample_shape))
        - sample_shape: a 3-tuple containing the size of each sample in
            train_data
        - latent_dimensions: the number of latent dimensions used by the
            VectorQuantizer layer in the VQVAE model
        - number_of_embeddings: the number of embeddings used by the
            VQVAE model
    '''
    vqvae = model.create_vqvae_model(
            latent_dimensions, number_of_embeddings, sample_shape)
    vqvae.compile(loss=vq_vae_loss(np.var(train_data)), optimizer="adam")
    return vqvae.fit(
        train_data, train_data, 
        validation_split=VQVAE_VALIDATION_SPLIT, 
        batch_size=VQVAE_BATCH_SIZE, epochs=VQVAE_EPOCHS
    ), vqvae

def test_vqvae(vqvae_model: keras.models.Sequential, test_data):
    '''
    Test the VQ VAE model and return relavent stats

    Parameters:
        - vqvae_model: A trained VQVAE model
        - test_data: Test data with a shape that the VQVAE model can accept
    '''
    recreated = vqvae_model.predict(test_data, use_multiprocessing=True)

    total_ssim = 0
    for i in range(test_data.shape[0]):
        total_ssim += ssim(recreated[i], test_data[i], multichannel=True)
    recreated_ssim = total_ssim / test_data.shape[0]

    return recreated, recreated_ssim

def train_pixel_cnn(x, y, latent_width, latent_height, number_of_embeddings):
    '''
    Trains the PixelCNN used for latent code generation for the decoder.
    '''
    print("Training PixelCNN for image generation")
    pixel_cnn = model.create_pixel_cnn(latent_width, latent_height, 
            number_of_embeddings)
    pixel_cnn.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
    history = pixel_cnn.fit(x, y, 
            batch_size=PIXELCNN_BATCH_SIZE, 
            epochs=PIXELCNN_EPOCHS, 
            validation_split=PIXELCNN_VALIDATION_SPLIT)

    print("Training complete.")

    return history, pixel_cnn

def generate_using_pixel_cnn(pixel_cnn_model, vqvae_decoder_model,
        vector_quantizer, number_of_images, quantized_shape):
    '''
    Generate images using the trained PixelCNN model and VQVAE decoder and
    vector quantizer layer. Generates the specified number_of_images and
    returns them.
    '''
    print(f"Generating {NUM_SAMPLE_IMAGES_GENERATED} images...")
    # Create priors by sampling the PixelCNN model
    priors = np.zeros(shape=(number_of_images,)
            + (pixel_cnn_model.input_shape)[1:])
    batch, rows, cols = priors.shape
    for row in range(rows):
        for col in range(cols):
            logits = pixel_cnn_model(priors, training=False)
            next_sample = tfp.distributions.Categorical(logits=logits).sample()
            priors[:, row, col] = next_sample.numpy()[:, row, col]

    # Use the priors to create latent space encodings
    one_hot_priors = tf.one_hot(priors.astype("int32"),
            vector_quantizer.number_of_embeddings).numpy()
    quantized = tf.matmul(one_hot_priors.astype("float32"),
        vector_quantizer.embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, (-1, *(quantized_shape)))

    # Generate samples from the latent space
    generated_samples = vqvae_decoder_model.predict(quantized)
    print("Completed generation")

    return generated_samples

def save_generated_samples(generated_samples, number_of_images):
    '''
    Save generated_samples both separately in the specified directory and as a 
    summary of 20 images in a 4 x 5 grid at generated_summary.
    '''
    print("Saving samples")
    MAX_SUMMARY = 20
    for i in range(MAX_SUMMARY):
        plt.subplot(4, 5 , i + 1)
        plt.imshow(generated_samples[i].squeeze())
        plt.axis("off")
    plt.title("Generated Brain MRIs (using VQ-VAE and PixelCNN)")
    plt.savefig("generated_summary.png")
    plt.close()
    print("Saved summary")
    print("Saving full resolution images...")
    
    try:
        os.mkdir(GENERATED_IMAGE_DIR)
    except FileExistsError:
        pass

    for i in range(number_of_images):
        file_path = os.path.join(GENERATED_IMAGE_DIR, f"{i}.png")
        keras.preprocessing.image.save_img(file_path, 
                generated_samples[i].squeeze())
    print(f"Save complete. See {GENERATED_IMAGE_DIR} and generated_summary.png")

# Loss Calculation Helpers

def vq_vae_loss(variance):
    '''
    Calculate the MSE loss with variance for the VQVAE model
    '''
    def calc_loss(x1, x2,):
        return tf.reduce_mean((x1 - x2) ** 2) / variance

    return calc_loss

def main():
    run_oasis()

if __name__ == "__main__":
    main()
