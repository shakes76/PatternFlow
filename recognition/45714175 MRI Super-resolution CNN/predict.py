import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def predict(model, image, epoch):
    """Use the model to predict the image from the lowres image and plot results"""
    # input image is downscaled test image
    input = tf.expand_dims(image, axis=0)
    output = model.predict(input)

    imageArray = tf.keras.preprocessing.image.img_to_array(output)
    imageArray = imageArray.astype("float32") / 255.0

    # Create a new figure for the epoch display images.
    fig, (ax1, ax2, ax3) = plt.subplots()
    fig.subtitle("Epoch: " + epoch)
    ax1.imshow(image)
    ax1.set_title("Downsampled Image")
    ax2.imshow(input)
    ax2.set_title("Downsampled Image")
    ax3.imshow(output)
    ax3.set_title("Downsampled Image")
    fig.show()
