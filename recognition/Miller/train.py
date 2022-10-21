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

# Load the validaton data from the oasis Data set 
validate_X = data.load_training ("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_validate")

# Pre process validation data set
validate_X = data.process_training(validate_X)

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
validate_Y = data.load_labels("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_seg_validate")
# Pre process validation labels data
validate_Y = data.process_labels(validate_Y)
 
# Load the segmented test labels data from the Oasis Data set
test_Y = data.load_labels("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_seg_test")
# Pre process test labels data
test_Y = data.process_labels(test_Y)

""" MODEL AND TRAIN VQ-VAE """
# Create a instance of the VQ-VAE model
latent_dimensions = 32 #dimensionality if each latent embedding vector
embeddings_number = 128 #number of embeddings in the codebook
#variance = np.var(train_X / 255.0)
model = mod.VQVAETRAINER(1, latent_dimensions, embeddings_number)

"""
Optimiser -> learning rate
'adam' adjusts learning rate whilst training; learning rate deterines how fast optimal weights are calculated. Smaller
Learning rate = more wights but takes longer to compute
"""
# Create Model
model.compile (optimizer='adam')

# Train model
history = model.fit(train_X, epochs=2, validation_data=(test_X), batch_size=128)
print("disaster!!!!")


# Plot Accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5,1])
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# evaluate against testing data 
test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=2)

print("Accuracy test is: ", test_acc)



""" MODEL AND TRAIN VQ-VAE """

""" RECONSTRUCTION RESULTS"""
# Plots the original image against the reconstructed one
def plot_comparision_original_to_reconstructed(original, reconstructed):
    plt.figure(figsize = (10,12))
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5, cmap = 'gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5, cmap = 'gray')
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()

trained_model = mod.model.model
idx = np.random.choice(len(test_X), 10)
test_images = test_X[idx]
reconstructions_test = trained_model.predict(test_images)

for test_image, reconstructed_image in zip(test_images, reconstructions_test):
    plot_comparision_original_to_reconstructed(test_image, reconstructed_image)

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
    
    image_variance = np.sqrt(image_variance/256**2 - 1)
    reconstructed_image_variance = np.sqrt(reconstructed_image_variance/256**2 - 1)
    return image_variance, reconstructed_image_variance

# Returns the covariance for both images
def calculate_covariance(image, reconstructed_image, image_mean, predicted_mean):
    covariance_value = 0
  
    for row in range(256):
        for col in range(256): 
            covariance_value += (image[row][col] - image_mean)*(reconstructed_image[row][col] - predicted_mean)
    
    return covariance_value/256**256-1


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

# Returns the structured similarity for the entire data set
def structural_similarity_mean(test_X, model):
    structured_similarity_coef = 0

    for i, data in enumerate(test_X):
        # get reconstructed image
        image_reconstruction = model.predict(data)
        data = data[0,:,:,0]
        image_reconstruction = image_reconstruction[0,:,:,0]

        # Calculate structured similarity and add to total
        mean_X, predicted_mean = calculate_mean(data, image_reconstruction)
        stddev_X, predicted_stddev = calculate_stddev(data, image_reconstruction, mean_X, predicted_mean)
        covariance = calculate_covariance(data, image_reconstruction, mean_X, predicted_mean)
        structured_similarity_coef += structural_similarity(mean_X, predicted_mean, stddev_X, predicted_stddev, covariance)

    return structured_similarity_coef / len(test_X)

print(structural_similarity_mean(test_X, trained_model))
# %%
