"""
COMP3170
Jialiang Hou
45996216
predict the test samples and report accuracy
"""
from keras.models import load_model
import random
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import dataset
import tensorflow.keras.backend as K
import numpy as np


def prediction(model, image, known_images, show = True):
    """
    use the trained model to predict a single image weather the brain has AD
    :param image: image need to predict
    :param known_images: the image we know label in format:[AD,NC]
    :param show: whether to show this image
    :return: the predicted label
    """
    result = model.predict([np.array([known_images[0], known_images[1]]), np.array([image, image])])
    if show:
        plt.imshow(image)
        if result[0] > result[1]:
            plt.title("This brain has Alzheimer's disease")
        else:
            plt.title("This brain does not have Alzheimer's disease")
        plt.show()
    if result[0] > result[1]:
        return 1
    else:
        return 0


def process_image(x):
    """
    input the image path return the image and process it
    :param x: the path of images
    :return: processed images
    """
    image = tf.io.read_file(x)
    image = tf.image.decode_png(image, channels=1)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(image, (60, 64))
    image = image / 255.0
    return image


model = load_model('./my_model.h5', custom_objects={'K': tf.keras.backend})
train_x1, train_x2, train_y, test_x1, test_x2, test_y = dataset.get_dataset()

# print the training accuracy
metrics = model.evaluate([train_x1, train_x2], train_y)
print('Loss of {} and Accuracy is {} %'.format(metrics[0], metrics[1] * 100))

# print the test accuracy
metrics = model.evaluate([test_x1, test_x2], test_y)
print('Loss of {} and Accuracy is {} %'.format(metrics[0], metrics[1] * 100))

image1 = process_image("./test/AD/392277_97.jpeg")
image2 = process_image("./test/NC/1188738_96.jpeg")
known_images = [process_image("./train/AD/250168_103.jpeg"),process_image("./train/NC/839474_87.jpeg")]

prediction(model,image1,known_images)
prediction(model,image2,known_images)

