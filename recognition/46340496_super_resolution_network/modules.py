import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from dataset import *
import PIL

# building the model
def get_model(upscale_factor=4, channels=1):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = tf.keras.Input(shape=(None, None, channels))
    x = tf.keras.layers.Conv2D(64, 5, **conv_args)(inputs)
    x = tf.keras.layers.Conv2D(64, 3, **conv_args)(x)
    x = tf.keras.layers.Conv2D(32, 3, **conv_args)(x)
    x = tf.keras.layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return tf.keras.Model(inputs, outputs)

# plotting results with zoom in area
def plot_results(img, prefix, title):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array.astype("float32") / 255.0

    # Create a new figure with a default 111 subplot.
    fig, ax = plt.subplots()
    im = ax.imshow(img_array[::-1], origin="lower")

    plt.title(title)
    # zoom-factor: 2.0, location: upper-left
    axins = zoomed_inset_axes(ax, 2, loc=2)
    axins.imshow(img_array[::-1], origin="lower")

    # Specify the limits.
    x1, x2, y1, y2 = 200, 300, 100, 200
    # Apply the x-limits.
    axins.set_xlim(x1, x2)
    # Apply the y-limits.
    axins.set_ylim(y1, y2)

    plt.yticks(visible=False)
    plt.xticks(visible=False)

    # Make the line.
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")
    plt.savefig(str(prefix) + "-" + title + ".png")
    plt.show()

# Getting the low-resolution image
def get_lowres_image(img):
    image = tf.image.resize(img, (256, 240))
    plt.imshow(image)
    plt.show()
    return image