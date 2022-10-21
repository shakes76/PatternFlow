import matplotlib.pyplot as plt
import numpy as np
from dataset import *
from modules import *
import tensorflow_probability as tfp

def codebook_wrapper_fn(encoder, embeddings):
    """
    The function encodes the images into codebook indices.

    param:
    encoder - Trained encoder from the VQVAE model
    embeddings - Trained latent codebook layer from VQVAE model

    Returns: Returns a mapper function so it can be applied to a
    dataset loader, so that all codebooks do not need to be stored in memory for training
    """

    def mapper(x):
        # Get the encoded outputs and flatten them
        encoded_outputs = encoder(x)

        flat_enc_outputs = tf.reshape(
            encoded_outputs, [-1, tf.shape(encoded_outputs)[-1]])

        codebook_indices = embeddings.get_closest_index(flat_enc_outputs)
        # Find nearest codebook indices
        codebook_indices = tf.reshape(
            codebook_indices, tf.shape(encoded_outputs)[:-1])

        return codebook_indices

    return mapper

def find_data_var(train_ds):
    """
    For the VQVAE model, the variance of the training dataset is required. Calculates the variance of the
    training dataset

    param:
    train_ds - Data loader for the training set

    Returns: Variance of the training dataset
    """
    # Find the mean of the data
    data_mean = 0
    number_data = 0
    for batch in train_ds:
        number_data += len(batch)
        data_mean += tf.reduce_sum(batch)

    data_mean = data_mean / (number_data * img_height * img_width * 3)

    data_var = 0
    # Find the variance of the data
    for batch in train_ds:
        data_var += tf.reduce_sum((batch - data_mean) ** 2)

    data_var = data_var / ((number_data * img_height * img_width * 3) - 1)
    return data_var

def training_plots(history):
    """
    Plots the training error and mean ssim of the validation data for the VQVAE model

    param:
    - History of the error in training
    """
    plt.plot(history.history['reconstruction_loss'], label="train")
    plt.plot(history.history['val_reconstruction_loss'], label="val")
    plt.title('Training Reconstruction Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

    plt.plot(history.history['val_mean ssim'], label="val")
    plt.title('Mean SSIM')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()

def pixel_cnn_training_plots(history2):
    """
    Plots the training error for the PixelCNN model

    param:
    - History of the error in training
    """
    plt.plot(history2.history['loss'], label="train")
    plt.plot(history2.history['val_loss'], label="val")
    plt.title('PixelCNN Sparse Categorical CrossEntropy Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()


# Constants / Hyperparameters
val_split = 0.2
img_height = 256
img_width = 256
batch_size = 32
latent_dim = 128 # Keep larger for more dimensions in the latent space
num_embeddings= 32 # Keep low so it is easier to train the pixelCNN

# Path to the training and test data. Currently for ADNI dataset
train_path = "AD_NC/train" # Path to the training data directory
test_path = "AD_NC/test" # Path to the test data directory

# Load in the training, validation and test data
train_ds = load_train_data(train_path, img_height, img_width, batch_size, val_split)
val_ds = load_validation_data(train_path, img_height, img_width, batch_size, val_split)
test_ds = load_test_data(test_path, img_height, img_width, batch_size)

data_var = find_data_var(train_ds) # Find the variance of the training data

# Initialising and compiling the model
vqvae_model = VQVAEModel(img_shape = (img_height, img_width, 3), embedding_num=num_embeddings, embedding_dim=latent_dim, beta=0.25, data_variance=data_var)
vqvae_model.compile(optimizer=keras.optimizers.Adam())
device = "/GPU:0" if len(tf.config.list_physical_devices('GPU')) else '/CPU:0'

# Training the model
with tf.device("/GPU:0"):
    history = vqvae_model.fit(train_ds, epochs=30, validation_data=val_ds, batch_size=batch_size)

# Saving the weights. Put the path of where you want to store the weights as input
vqvae_model.save_weights("path to store weights")

# Plotting code and results
training_plots(history)

# Training the PixelCNN.
# Load in the data
pixel_train_ds = load_train_data(train_path, img_height, img_width, batch_size, 0.8)
pixel_val_ds = load_validation_data(train_path, img_height, img_width, batch_size, 0.05)
codebook_mapper = codebook_wrapper_fn(
        vqvae_model.get_encoder(),
        vqvae_model.get_vq())
codebook_dataset = pixel_train_ds.map(codebook_mapper)
codebook_val_dataset = pixel_val_ds.map(codebook_mapper)
pixelcnn_input_shape = vqvae_model.get_encoder().output.shape[1:3]

# Make and compile the model
pixelcnn_model = PixelCNNModel(pixelcnn_input_shape, vqvae_model._embedding_num, 128, 2,2)
pixelcnn_model.compile(optimizer=keras.optimizers.Adam(0.0003),
                       loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=["accuracy"]
)

# Training the model
with tf.device("/GPU:0"):
    history2 = pixelcnn_model.fit(codebook_dataset, batch_size=64, epochs=100, validation_data=codebook_val_dataset)

# Saving the weights. Put the path of where you want to store the weights as input
pixelcnn_model.save_weights("path to store weights")

# Training error plots
pixel_cnn_training_plots(history2)

