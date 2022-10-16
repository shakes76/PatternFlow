import os
from matplotlib import test
import tensorflow as tf
# import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tensorflow.keras import datasets, layers, models

data_path = "../../../ISIC_Data/ISIC-2017_Training_Data/*.jpg"
mask_path = "../../../ISIC_Data/ISIC-2017_Training_Part1_GroundTruth/*.png"

# new_img = tf.io.read_file("./TestImages/ISIC_0000000.jpg")
# raw_image = tf.io.decode_jpeg(new_img, channels=3)


def prepareData(filenames):
    new_img = tf.io.read_file(filenames)
    raw_image = tf.io.decode_jpeg(new_img, channels=3)

    # Resize
    raw_image = tf.image.resize_with_pad(raw_image, 480, 480)

    # Normalise
    raw_image = raw_image / 255.0

    return raw_image


def prepareMasks(filenames):
    new_img = tf.io.read_file(filenames)
    raw_image = tf.io.decode_png(new_img, channels=3)

    # Resize
    raw_image = tf.image.resize_with_pad(raw_image, 480, 480)

    # Normalise
    raw_image = raw_image / 255.0

    # Set image thresholds
    raw_image = tf.where(raw_image > 0.5, 1.0, 0.0)

    return raw_image


ISIC_Data = tf.data.Dataset.list_files(data_path, shuffle=False)
preData = ISIC_Data.map(prepareData)

Mask_Data = tf.data.Dataset.list_files(mask_path, shuffle=False)
preMasks = Mask_Data.map(prepareMasks)

# for i in preData.take(1):
#     plt.imshow(i.numpy())
#     plt.show()

# for j in preMasks.take(1):
#     plt.imshow(j.numpy())
#     plt.show()

im = tf.io.read_file("../../../ISIC_Data/ISIC-2017_Training_Data/ISIC_0000000.jpg")
box = np.array([0.1, 0.2, 0.5, 0.9])
boxes = box.reshape([1, 1, 4])
colours = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
bbox = tf.image.draw_bounding_boxes(im, boxes, colours)

# plt.figure(figsize=(1022, 767))
# plt.imshow(images[i].numpy().astype("uint8"))
# plt.title(class_names[labels[i]])
# plt.axis("off")
# # testTensor = tf.constant(4)
# # # img = tf.io.decode_png(training_path)

# # print(testTensor)