import os
import tensorflow as tf
import numpy as np

from tensorflow import keras

from dataset import *
from modules import *
from train import *

from tensorflow.keras.preprocessing.image import load_img, img_to_array

REQ_DIR = "../content/drive/MyDrive/Colab Notebooks/Report/AD_NC"

def get_test_data(dir):
    AD_test_images = os.listdir(dir + "/test/AD")

    test_both = []

    for image_name in AD_test_images[:20]:
        image = load_img(dir + "/test/AD/" + image_name, target_size = (128, 128, 3))
        image = img_to_array(image)
        test_both.append([image,1])
    
    test_images = []
    test_labels = []

    for x in test_both:
        test_images.append(x[0])
        test_labels.append(x[1])

    x_test = tf.convert_to_tensor(np.array(test_images, dtype=np.uint8))
    x_test = tf.cast(x_test, tf.float16) / 255.0
    y_test = tf.convert_to_tensor(test_labels)

    return x_test, y_test

def main():
    new_model = tf.keras.models.load_model('complete_model')
    x_test, y_test = get_test_data(REQ_DIR)
    loss, acc = new_model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {round(acc * 100, 2)}%")


if __name__ == "__main__":
    main()