import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from skimage.metrics import structural_similarity
from model import *

###############################################################
# Pre-processing of the OASIS brain data 
# FROM DEMO 2 - permision granted by tutor to use same code

# Download the entire OASIS dataset and save to local drive
dataset_url = "https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA/download"
data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='oasis_data', extract=True)

# Use the glob library to separate and save the train, test and validation datasets to local drive
train_images = glob.glob('C:/Users/s4447019/.keras/datasets/keras_png_slices_data/keras_png_slices_train' + '/*.png')
validate_images = glob.glob('C:/Users/s4447019\.keras\\datasets\\keras_png_slices_data\\keras_png_slices_validate' + '/*.png')
test_images = glob.glob('C:\\Users\\s4447019\\.keras\\datasets\\keras_png_slices_data\\keras_png_slices_test'  + '/*.png')

# Print the number of images available within each dataset (sanity check to ensure data is available)
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
plt.title("Sample of images from OASIS train dataset")
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(next(iter(train_ds.batch(9)))[i])
    plt.axis('off')
plt.show()
###############################################################

###############################################################
# Define the variance of the training images
# Note: Easiest solution to find variance of all training data
# was to parse entire training set as one 'batch'
data_variance = tf.image.total_variation(next(iter(train_ds.batch(9664)))) 

# Train the VQVAE model
vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=32) #Reduced num_embeddings to resolve memory errors
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
vqvae_trainer.fit(train_ds.batch(32), epochs=30) #Training for batch=32 and 30 epochs
###############################################################

###############################################################
# Reconstucted results on the test set
# Reference: https://keras.io/examples/generative/vq_vae/
# Author: Sayak Paul

# Print the test image and their reconstruction
trained_vqvae_model = vqvae_trainer.vqvae
test_images = next(iter(test_ds.batch(10)))
reconstructions_test = trained_vqvae_model.predict(test_images)

for test_image, reconstructed_image in zip(test_images, reconstructions_test):
    plt.subplot(1, 2, 1)
    plt.imshow(test_image.numpy().squeeze() + 0.5)
    plt.title("Original_image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image.squeeze() + 0.5)
    plt.title("Reconstructed_image")
    plt.axis("off")

    plt.show()
###############################################################        
    
###############################################################
# Code results on the test set
# Reference: https://keras.io/examples/generative/vq_vae/
# Author: Sayak Paul

# Print the test image and their associated coded image
encoder = vqvae_trainer.vqvae.get_layer("encoder")
quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

encoded_outputs = encoder.predict(test_images)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

for i in range(len(test_images)):
    plt.subplot(1, 2, 1)
    plt.imshow(test_images[i] + 0.5)
    plt.title("Original_image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(codebook_indices[i])
    plt.title("Code_image")
    plt.axis("off")
    
    plt.show()
###############################################################

###############################################################
# Implementing the PIXEL_CNN
# Reference: https://keras.io/examples/generative/vq_vae/
# Author: Sayak Paul

# Define the number of residual blocks and pixelcnn layers
num_residual_blocks = 2
num_pixelcnn_layers = 2
pixelcnn_input_shape = encoded_outputs.shape[1:-1]
print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")

pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
ohe = tf.one_hot(pixelcnn_inputs, vqvae_trainer.num_embeddings)
x = PixelConvLayer(
    mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same")(ohe)

for _ in range(num_residual_blocks):
    x = ResidualBlock(filters=128)(x)
for _ in range(num_pixelcnn_layers):
    x = PixelConvLayer(mask_type="B", filters=32, kernel_size=1, strides=1,
        activation="relu", padding="valid")(x)
out = keras.layers.Conv2D(
    filters=vqvae_trainer.num_embeddings, kernel_size=1, strides=1, padding="valid")(x)

pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")
pixel_cnn.summary() #Print a summary of the model
##############################################################

##############################################################
# Generatation of the codebook indices
encoded_outputs = encoder.predict(next(iter(train_ds.batch(9664)))) #Using entire training set here
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)

codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
#############################################################

#############################################################
# Compile and run the pixel_cnn model
pixel_cnn.compile(optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"])
pixel_cnn.fit(x=codebook_indices, y=codebook_indices, batch_size=128, epochs=40, validation_split=0.05) #Reduced split from tutorial so more images could be used for training
#############################################################

#############################################################
# Implementing the PIXEL_CNN (cont.)
# Reference: https://keras.io/examples/generative/vq_vae/
# Author: Sayak Paul

# To create a sampler model, discrete codes are sampled from PixelCNN outputs
# to be passed into the decoder to generate the novel images
inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
x = pixel_cnn(inputs, training=False)
dist = tfp.distributions.Categorical(logits=x)
sampled = dist.sample()
sampler = keras.Model(inputs, sampled)

# Creatation of an empty array of priors
batch = 10
priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:] + (32,))
batch, rows, cols, channels = priors.shape
print(f"Prior shape: {priors.shape}") #Check the shape before appending

# Iterate over the 2D array
for row in range(rows):
    for col in range(cols):
        probs = pixel_cnn.predict(priors)[:, row, col, :]
        # Retrive maximum probability value and append to priors
        priors[:, row, col] = tf.argmax(probs, axis=1)
        
# Perform an embedding lookup
pretrained_embeddings = quantizer.embeddings
priors_ohe = tf.one_hot(priors.astype("int32"), vqvae_trainer.num_embeddings).numpy()
quantized = tf.matmul(priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True)
quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))

# Generation of the novel images
# This does not work as it should - see above explanation
decoder = vqvae_trainer.vqvae.get_layer("decoder")
generated_samples = decoder.predict(quantized)

for i in range(batch):
    plt.subplot(1, 2, 1)
    plt.imshow(priors[i,0])
    plt.title("Code_image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(generated_samples[i].squeeze() + 0.5)
    plt.title("Generated Sample_image")
    plt.axis("off")
    plt.show()
#############################################################

#############################################################
# Reference : https://ourcodeworld.com/articles/read/991/how-to-calculate-the-structural-similarity-index-ssim-between-two-images-with-python
# Author: Carlos Delgado

# Compute the Structural Similarity Index (SSIM) between the two images for all test images
total_score = 0

for i in range(544): #544 is the number of test images
    test_images = next(iter(test_ds.batch(1)))
    reconstructions_test = trained_vqvae_model.predict(test_images)
    (score, diff) = structural_similarity(test_images.numpy()[0], reconstructions_test[0], full=True, multichannel=True)
    diff = (diff * 255).astype("uint8")
    total_score += score

total_score = total_score / 544 #Get the mean SSIM from all test images

# Print the mean SSIM
print("Total SSIM: {}".format(total_score))
#############################################################