"""
predict.py
Used for model predictions of downsampled images
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import display

def predict(model, image):
    """Use the model to predict the image from the lowres image and plot results"""
    # input image is downscaled test image
    input = tf.expand_dims(image, axis=0)
    output = model.predict(input)

    # Display predicted image
    display(tf.keras.preprocessing.image.array_to_img(output[0] / 255.0))
