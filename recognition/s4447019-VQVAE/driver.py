import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from skimage.metrics import structural_similarity


### PRE-PROCESSING FROM DEMO 2 ################################

# Download the entire OASIS dataset and save to local drive
dataset_url = "https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA/download"
data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='oasis_data', extract=True)

# Use the glob library to separate and save the train, test and validation datasets to local drive
train_images = glob.glob('C:\\Users\\mmene\\.keras\\datasets\\keras_png_slices_data\\keras_png_slices_train' 
                         + '/*.png')
validate_images = glob.glob('C:\\Users\\mmene\\.keras\\datasets\\keras_png_slices_data\\keras_png_slices_validate' + '/*.png')
test_images = glob.glob('C:\\Users\\mmene\\.keras\\datasets\\keras_png_slices_data\\keras_png_slices_test' + '/*.png')

# Print the number of images available within each dataset
print('Number of training images:', len(train_images))
print('Number of validation images:', len(validate_images))
print('Number of testing images:', len(test_images))

# Transform the image data into tensors for preprocessing
train_ds = tf.data.Dataset.from_tensor_slices(train_images)
validate_ds = tf.data.Dataset.from_tensor_slices(validate_images)
test_ds = tf.data.Dataset.from_tensor_slices(test_images)

# Reshuffle the data every epoch so that there are different batches for each epoch
# Note: buffer needs to be greater than, or equal to the size of the data set for effective shuffling
train_ds = train_ds.shuffle(len(train_images))
validate_ds = validate_ds.shuffle(len(validate_images))
test_ds = validate_ds.shuffle(len(test_images))

# Mapping the filenames to data arrays
#---Reference for function used below: 
#---COMP3710 lecture, 24th October 2020---#
def map_fn(filename):
    img = tf.io.read_file(filename) #Open the file
    img = tf.image.decode_png(img, channels=1) #Defining the number of channels (1 for B&W images)
    img = tf.image.resize(img, (128, 128)) #Resize the images to feed into the network
    img = tf.cast(img, tf.float32) / 255.0 #Normalise the data
    return img

# Update the datasets by passing them in map_fn()
train_ds = train_ds.map(map_fn)
validate_ds = validate_ds.map(map_fn)
test_ds = test_ds.map(map_fn)

# Print a sample of images from the test dataset
plt.figure(figsize=(10,10))
plt.title("Sample of images from OASIS test dataset")
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(next(iter(train_ds.batch(9)))[i])
    plt.axis('off')
plt.show()
###############################################################



###### VQ from Keras tutorial #################################
# Vector Quantizer class from Kera's tutorial
# Reference: https://keras.io/examples/generative/vq_vae/

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (beta)
        # The commitment loss coefficient, beta, is best kept between [0.25, 2]
        # as per Neural Discrete Representation Learning (page 4)

        # Initialisation of embeddings to be quantised
        w_init = tf.random_uniform_initializer() #Initialisation of random values
        self.embeddings = tf.Variable( #Definition of the embeddings tensor variable
            initial_value=w_init(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices
###############################################################


###### VQ Trainer from Keras tutorial #########################
# VQVAE Trainer class from Kera's tutorial
# Reference: https://keras.io/examples/generative/vq_vae/
# Author: Sayak Paul

class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=16, num_embeddings=128, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance #Define the variance of the training image set
        self.latent_dim = latent_dim #Define the latent dimension of the model
        self.num_embeddings = num_embeddings #Define the total number of embeddings of the model

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings) #Call the model with the given hyperparameters
        
        # Define the reconstuction and total loss variables to be printed during training
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # These are the outputs from the VQ-VAE
            reconstructions = self.vqvae(x)

            # Calculate the total and reconstruction losses
            reconstruction_loss = (tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance)
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Application of backpropagation
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Update the loss variables
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        # Print the loss results
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }
    
###############################################################

# Define the variance of the training images
data_variance = tf.image.total_variation(next(iter(train_ds.batch(9664)))) 
# Note: Easiest solution to find variance of all training data
# was to parse entire training set as one 'batch'

# Train the VQVAE model
vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=32)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
vqvae_trainer.fit(train_ds.batch(32), epochs=30) #Training for batch=32 and 30 epochs