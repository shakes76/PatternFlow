"""
“predict.py" showing example usage of your trained model. Print out any results and / or provide visualisations where applicable
"""
import numpy as np
import matplotlib.pyplot as plt
import modules as mod

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Show how well program performs 


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