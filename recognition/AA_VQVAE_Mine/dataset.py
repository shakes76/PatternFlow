import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

import os

image_height = 240
image_width = 256
b_size = 32

im_root = path = os.path.join(os.getcwd(), "recognition\AA_VQVAE_Mine\DataSets\AD_NC")


training_set = tf.keras.utils.image_dataset_from_directory(
                        os.path.join(im_root,"train"),
                        labels='inferred',
                        label_mode='int',
                        color_mode='grayscale',
                        image_size=(image_width, image_height),
                        batch_size = b_size,
                        shuffle=True,
                        seed=46,
                        validation_split=0.3,
                        subset='training',
                        interpolation='bilinear',
                        crop_to_aspect_ratio=True
                    )

validation_set = tf.keras.utils.image_dataset_from_directory(
                        os.path.join(im_root,"train"),
                        labels='inferred',
                        label_mode='int',
                        color_mode='grayscale',
                        image_size=(image_width, image_height),
                        batch_size = b_size,
                        shuffle=True,
                        seed=46,
                        validation_split=0.3,
                        subset='validation',
                        interpolation='bilinear',
                        crop_to_aspect_ratio=True
                    )

test_set = tf.keras.utils.image_dataset_from_directory(
                    os.path.join(im_root,"test"),
                    labels='inferred',
                    label_mode='int',
                    color_mode='grayscale',
                    image_size=(image_width, image_height),
                    batch_size = b_size,
                    shuffle=True,
                    seed=46,
                    interpolation='bilinear',
                    crop_to_aspect_ratio=True
                )

class_names = training_set.class_names
print(class_names)

'''
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
x_val = np.expand_dims(x_val, -1)
x_train_scaled = (x_train / 255.0) - 0.5
x_test_scaled = (x_test / 255.0) - 0.5
x_val_scaled = (x_val / 255.0) - 0.5

data_variance = np.var(x_train / 255.0)
'''


#And plot images
plt.figure(figsize=(10, 10))
for images, labels in training_set.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"),cmap='gray')
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()
