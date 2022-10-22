"""
predict.py
Used for model predictions of downsampled images
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import display
from constants import *

def predict(model, image):
    """Use the model to predict the image from the lowres image and plot results"""
    # input image is downscaled test image
    input = tf.expand_dims(image, axis=0)
    output = model.predict(input)

    # Display predicted image
    display(tf.keras.preprocessing.image.array_to_img(output[0] / 255.0))

def displayPredictions(model, testData):
    """Function used to display the Original image, the low resolution image
        and the upscaled image the model has predicted"""
    for image in testData.take(5):
        original = tf.keras.preprocessing.image.array_to_img(image[0])
        downscaledImage = tf.image.resize(image[0], 
                            (HEIGHT // DOWNSCALE_FACTOR, WIDTH // DOWNSCALE_FACTOR), 
                            method="gaussian")

        predict(model, downscaledImage)
        print("[Upscaled Image]")
        
        display(original)
        print("[Original Image]")