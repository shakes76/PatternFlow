import pathlib
import tensorflow as tf

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

# Creating the test dataset
def creating_test_dataset():

    test_ds  = tf.keras.utils.image_dataset_from_directory(
        import_test_data(),
        batch_size=BATCH_SIZE,
        image_size=(HEIGHT, WIDTH),
        seed=1337,
        label_mode=None,
        color_mode="grayscale",
    )
    return test_ds

# Scale from (0, 255) to (0, 1)
def scaling(input_image):
    input_image = input_image / 255.0
    return input_image

# Rescaling the datasets
def mapping():
    train_ds_raw, valid_ds_raw = creating_train_datasets()

    train_ds = train_ds_raw.map(scaling)
    valid_ds = valid_ds_raw.map(scaling)

    return train_ds, valid_ds

# Mapping the low-resolution dataset to the target dataset
def mapping_target():

    train_ds, valid_ds = mapping()

    train_ds = train_ds.map(
        lambda x: (tf.image.resize(x, (64, 60)), x)
    )

    valid_ds = valid_ds.map(
        lambda x: (tf.image.resize(x, (64, 60)), x)
    )

    return train_ds, valid_ds

