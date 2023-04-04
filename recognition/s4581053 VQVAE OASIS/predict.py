"""
“predict.py" showing example usage of your trained model. Print out any results and / or provide visualisations where applicable
"""
import numpy as np
import matplotlib.pyplot as plt
import modules as mod
import dataset as data

""" MODEL AND TRAIN VQ-VAE """
# Load the training data from the Oasis Data set
train_X = data.load_training ("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_train")
train_X = data.process_training(train_X)
train_x_var = np.var(train_X)

# Load the test data from the oasis Data Set 
test_X = data.load_training ("C:/Users/dapmi/OneDrive/Desktop/Data/oa-sis.tar/keras_png_slices_data/keras_png_slices_test")

# Pre process test data set
test_X = data.process_training(test_X)

""" RECONSTRUCTION RESULTS"""
latent_dimensions = 16 #dimensionality if each latent embedding vector
embeddings_number = 128 #number of embeddings in the codebook
# load model
model = mod.VQVAETRAINER(train_x_var, latent_dimensions, embeddings_number)
# Create Model
model.compile (optimizer='adam')

# Train model
history = model.fit(train_X, epochs=5, batch_size=128)

# Plot Loss
plt.plot(history.history['reconstruction_loss'], label='Reconstruction Loss')
plt.title('VQVAE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

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

trained_model = model.vqvae_model

# Select 5 random Test images
idx = np.random.choice(len(test_X), 5)
test_images = test_X[idx]
reconstructions_test = trained_model.predict(test_images)

# Perform Predictions on the test images
for test_image, reconstructed_image in zip(test_images, reconstructions_test):
   
    plot_comparision_original_to_reconstructed(test_image, reconstructed_image)

