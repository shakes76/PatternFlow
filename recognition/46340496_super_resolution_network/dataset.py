import pathlib
import tensorflow as tf
import os

import matplotlib.pyplot as plt

HEIGHT = 256
WIDTH = 240
BATCH_SIZE = 8
UPSCALE_FACTOR = 4
# INPUT_SIZE = HEIGHT // UPSCALE_FACTOR # NEED TO FIGURE THIS OUT

# import the data
def import_train_data():
    train_dir_str = r"C:\Users\galla\OneDrive\University\Year 3\Semester 2\COMP3710\Report\AD_NC\train\AD"
    train_dir = pathlib.Path(train_dir_str)
    return train_dir

def import_test_data():
    test_dir_str = r"C:\Users\galla\OneDrive\University\Year 3\Semester 2\COMP3710\Report\AD_NC\test\AD"
    test_dir = pathlib.Path(test_dir_str)
    return test_dir

# Creating training and validation datasets
def creating_train_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        import_train_data(),
        batch_size=BATCH_SIZE,
        image_size=(HEIGHT, WIDTH),
        validation_split=0.2,
        subset="training",
        seed=1337,
        label_mode=None,
        color_mode="grayscale",
    )

    print("initial tarin_ds", train_ds)

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        import_train_data(),
        batch_size=BATCH_SIZE,
        image_size=(HEIGHT, WIDTH),
        validation_split=0.2,
        subset="validation",
        seed=1337,
        label_mode=None,
        color_mode="grayscale",
    )

    return train_ds, valid_ds

def creating_test_dataset():

    test_path = import_test_data()

    test_ds = sorted(
        [
            os.path.join(test_path, fname)
            for fname in os.listdir(test_path)
            if fname.endswith(".jpeg")
        ]
    )
    return test_ds

# Rescaling
def scaling(input_image):
    input_image = input_image / 255.0
    return input_image


# Scale from (0, 255) to (0, 1)
def mapping():
    train_ds_raw, valid_ds_raw = creating_train_datasets()

    train_ds = train_ds_raw.map(scaling) # normalization layers for scaling maybe tf.rescaling
    valid_ds = valid_ds_raw.map(scaling)

    return train_ds, valid_ds

# train_ds, valid_ds = mapping()
test_ds = creating_test_dataset()


# #Visualising images
# for batch in train_ds.take(1):
#     for img in batch:
        
#         plt.imshow(img.numpy())
#         # plt.show()

# # converting images from RGB to YUV from the low-resolution images
# def process_input(input, input_size):
#     input = tf.image.rgb_to_yuv(input)
#     last_dimension_axis = len(input.shape) - 1
#     y, u, v = tf.split(input, 3, axis=last_dimension_axis)
#     return tf.image.resize(y, [input_size, input_size], method="area")

# # converting images from RBG to YUB from the high-resolution images
# def process_target(input):
#     input = tf.image.rgb_to_yuv(input)
#     last_dimension_axis = len(input.shape) - 1
#     y, u, v = tf.split(input, 3, axis=last_dimension_axis)
#     return y

def mapping_target():

    train_ds, valid_ds = mapping()

    print("train_ds", train_ds)

    train_ds = train_ds.map(
        lambda x: (tf.image.resize(x, [64, 60]), x)
        )

    train_ds = train_ds.prefetch(buffer_size=32)

    valid_ds = valid_ds.map(
        lambda x: (tf.image.resize(x, [64, 60]), x)
        )

    valid_ds = valid_ds.prefetch(buffer_size=32)

    return train_ds, valid_ds

# Visualise input and target
# for batch in train_ds.take(1):
#     for img in batch[0]:
#         img_plot = plt.imshow(img)
#     for img in batch[1]:
#         img_plot = plt.imshow(img)
