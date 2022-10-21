"""
“train.py" containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
"""
# %%
import dataset as data
import modules as mod
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Download Data and then unzip
#download_oasis()
# %%

""" PROCESS TRAINING DATA"""
# Load the training data from the Oasis Data set
train_X = data.load_training ("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_train")

# Check training image
#pyplot.imshow(train_X[2])
#pyplot.show()

# Pre process training data set
train_X = data.process_training(train_X)
train_x_var = np.var(train_X)
# Load the validaton data from the oasis Data set 
#validate_X = data.load_training ("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_validate")

# Pre process validation data set
#validate_X = data.process_training(validate_X)

# Load the test data from the oasis Data Set 
test_X = data.load_training ("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_test")

# Pre process test data set
test_X = data.process_training(test_X)

""" PROCESS TRAINING LABELS DATA """
# Load the segmented training labels data from the Oasis Data set
train_Y = data.load_labels ("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_seg_train")
# Pre process training labels data
train_Y = data.process_labels(train_Y)

# Load the segmented validation labels data from the Oasis Data set
#validate_Y = data.load_labels("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_seg_validate")
# Pre process validation labels data
#validate_Y = data.process_labels(validate_Y)
 
# Load the segmented test labels data from the Oasis Data set
test_Y = data.load_labels("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_seg_test")
# Pre process test labels data
test_Y = data.process_labels(test_Y)
#%%
""" MODEL AND TRAIN VQ-VAE """
# Create a instance of the VQ-VAE model
latent_dimensions = 16 #dimensionality if each latent embedding vector
embeddings_number = 128 #number of embeddings in the codebook

model = mod.VQVAETRAINER(train_x_var, latent_dimensions, embeddings_number)

"""
Optimiser -> learning rate
'adam' adjusts learning rate whilst training; learning rate deterines how fast optimal weights are calculated. Smaller
Learning rate = more wights but takes longer to compute
"""
# Create Model
model.compile (optimizer='adam')

# Train model
history = model.fit(train_X, epochs=5, batch_size=128)
print("disaster!!!!")


#%%
# Plot Loss
plt.plot(history.history['reconstruction_loss'], label='Reconstruction Loss')
plt.plot(history.history['loss'], label='Reconstruction Loss')
plt.plot(history.history['vqvae_loss'], label='Reconstruction Loss')
plt.title('VQVAE Reconstruction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training Data', 'Validation'], loc='upper right')
plt.show()

#%%

""" MODEL AND TRAIN VQ-VAE """

""" RECONSTRUCTION RESULTS"""
# Return the average pixel value for the image and the reconstruction
def calculate_mean(image, reconstructed_image):
    image_pixel = 0
    reconstructed_pixel = 0

    for row in range(256):
        for col in range(256):
            image_pixel += image[row][col]
            reconstructed_pixel += reconstructed_image[row][col]

    image_pixel = image_pixel / (256**2)
    reconstructed_pixel = reconstructed_pixel / (256**2)

    return image_pixel, reconstructed_image

# Returns std dev for the pixel value of each image
def calculate_stddev(image, reconstructed_image, image_mean, reconstructed_image_mean):

    image_variance = 0
    reconstructed_image_variance = 0

    for row in range(256):
        for col in range(256):
            image_variance += np.square(image[row][col] - image_mean)
            reconstructed_image_variance += np.square(reconstructed_image[row][col] - reconstructed_image_mean)
    
    image_variance = np.sqrt(image_variance/(256**2 - 1))
    reconstructed_image_variance = np.sqrt(reconstructed_image_variance/(256**2 - 1))
    return image_variance, reconstructed_image_variance

# Returns the covariance for both images
def calculate_covariance(image, reconstructed_image, image_mean, predicted_mean):
    covariance_value = 0
  
    for row in range(256):
        for col in range(256): 
            covariance_value += (image[row][col] - image_mean)*(reconstructed_image[row][col] - predicted_mean)
    
    return covariance_value/(256**256-1)


# Return the structural similarity between two images; measures the window x and y of common size.
# https://en.wikipedia.org/wiki/Structural_similarity
def structural_similarity(mean_X, predicted_mean, stddev_X, predicted_stddev, covariance):
    K1 = 0.01 # default value
    K2 = 0.03 # default value
    L = 255 # dynamic range of pixel value (2^bits per pixel -1)
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2
    C3 = C2 / 2
    
    luminance_x_y = (2*mean_X*predicted_mean + C1)/(mean_X**2+predicted_mean**2+C1)
    contrast_x_y = (2*stddev_X*predicted_stddev + C2)/(stddev_X**2+np. predicted_stddev**2+C2)
    structure_x_y = (covariance+C3)/(stddev_X*predicted_stddev+C3)
    return luminance_x_y * contrast_x_y * structure_x_y

# Plots the original image against the reconstructed one with their Structured similarity rating
def plot_comparision_original_to_reconstructed(original, reconstructed, ssim):
    plt.suptitle("Structured Similiarity Rating: %.2f" %ssim)

    #plt.figure(figsize = (10,12))
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5, cmap = 'gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5, cmap = 'gray')
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()

trained_model = model.vqvae_model

# Select 5 random Test images
idx = np.random.choice(len(test_X), 5)
test_images = test_X[idx]
reconstructions_test = trained_model.predict(test_images)
ssim_mean = 0
# Perform Predictions on the test images
for test_image, reconstructed_image in zip(test_images, reconstructions_test):
    """mean, mean_r = calculate_mean(test_image, reconstructed_image)
    stddev, stddev_r = calculate_stddev(test_image,reconstructed_image, mean, mean_r)
    cov = calculate_covariance(test_image, reconstructed_image, mean, mean_r)
    structured_similiarity_rating = structural_similarity(mean, mean_r, stddev, stddev_r, cov)
    """
    structured_similiarity_rating = tf.image.ssim(test_image, reconstructed_image, max_val=1.0)
    plot_comparision_original_to_reconstructed(test_image, reconstructed_image, structured_similiarity_rating)
    ssim_mean += structured_similiarity_rating
    print(structured_similiarity_rating)

# Calculate the mean structural Similarity for the reconstructed images
print("The average structured similiarity rating is: ", ssim_mean/len(test_images))


"""Visualizing the discrete codes"""
"""
encoder = model.vqvae_model.get_layer("encoder")
quantizer = model.vqvae_model.get_layer("vector_quantizer")

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

"""

""" PIXELCNN Hyperparameters"""
"""
residualblock_num = 2
pixelcnn_layers = 2
pixelcnn_input_shape = encoded_outputs.shape[1:-1]
print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")


pixel_model = mod.pixel_model(pixelcnn_input_shape, residualblock_num, pixelcnn_layers, model)

"""

""" DATA PREPARATION"""
"""
# Generate the codebook indices.
codebook_indices = data.codebook_indice_generator(train_X, encoder, quantizer)

"""
""" PixelCNN TRAINING"""
"""
pixel_model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
pixel_model.fit(
    x=codebook_indices,
    y=codebook_indices,
    batch_size=128,
    epochs=5,
    validation_split=0.1,
)

# Create a mini sampler model.
inputs = tf.keras.layers.Input(shape=pixel_model.input_shape[1:])
outputs = pixel_model(inputs, training=False)
categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
outputs = categorical_layer(outputs)
sampler = tf.keras.Model(inputs, outputs)

# Create an empty array of priors.
batch = 10
priors = np.zeros(shape=(batch,) + (pixel_model.input_shape)[1:])
batch, rows, cols = priors.shape

# Iterate over the priors because generation has to be done sequentially pixel by pixel.
for row in range(rows):
    for col in range(cols):
        # Feed the whole array and retrieving the pixel value probabilities for the next
        # pixel.
        probs = sampler.predict(priors)
        # Use the probabilities to pick pixel values and append the values to the priors.
        priors[:, row, col] = probs[:, row, col]

print(f"Prior shape: {priors.shape}")

# Perform an embedding lookup.
pretrained_embeddings = quantizer.embeddings
priors_ohe = tf.one_hot(priors.astype("int32"), model.num_embeddings).numpy()
quantized = tf.matmul(
    priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
)
quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))

# Generate novel images.
decoder = model.vqvae.get_layer("decoder")
generated_samples = decoder.predict(quantized)

for i in range(batch):
    plt.subplot(1, 2, 1)
    plt.imshow(priors[i])
    plt.title("Code")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(generated_samples[i].squeeze() + 0.5)
    plt.title("Generated Sample")
    plt.axis("off")
    plt.show()

    """