"""
“predict.py" showing example usage of your trained model. Print out any results and / or provide visualisations where applicable
"""
import tensorflow as tf 
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import glob
import modules as mod
import train as t
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Show how well program performs 


""" MODEL AND TRAIN VQ-VAE """
# Create a instance of the VQ-VAE model
latent_dimensions = 16
embeddings_number = 64
image_size = 256
# beta = [0.25, 2]
beta = 0.25
model = mod.vqvae_model(image_size, latent_dimensions, embeddings_number, beta)

model.summary()


model.compile (optimizer="Adam", loss= tf.keras.losses.CategoricalCrossentropy())

# record history of training to display loss over ephocs 
history = model.fit(t.train_X, t.train_Y,  validation_data= (t.validate_X, t.validate_Y) ,batch_size=32,shuffle=True,epochs=5)

# evaluate against testing data 
model.evaluate(t.test_X,t.test_Y)

def show_subplot(original, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()



idx = np.random.choice(len(t.test_X), 10)
test_images = t.test_X[idx]
reconstructions_test = model.predict(test_images)

for test_image, reconstructed_image in zip(test_images, reconstructions_test):
    show_subplot(test_image, reconstructed_image)