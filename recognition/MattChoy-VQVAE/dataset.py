from constants import image_shape, batch_size, dataset_seed

import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

dataset_fp = os.path.join("data", "OASIS")

x_train = image_dataset_from_directory(
    os.path.join(dataset_fp, "train"),
    batch_size=batch_size,
    image_size=image_shape,
    seed=dataset_seed,
    label_mode = None
)

x_test = image_dataset_from_directory(
    os.path.join(dataset_fp, "test"),
    batch_size=batch_size,
    image_size=image_shape,
    seed=dataset_seed,
    label_mode = None
)

def scaling(input_image):
    input_image = input_image / 255.0
    # input_image = tf.image.rgb_to_grayscale(input_image)
    return input_image

x_train = x_train.map(scaling)
x_test = x_test.map(scaling)
