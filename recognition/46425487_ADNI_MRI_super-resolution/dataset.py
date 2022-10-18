import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory

from IPython.display import display
from matplotlib import pyplot as plt

def process_data():

    train_ds= image_dataset_from_directory(
        r"D:\Everything\Uni\2022\Sem2\COMP3710\report\AD_NC\train",
        image_size=(240, 256),
        label_mode=None,
        color_mode = "grayscale"
    )

    valid_ds = image_dataset_from_directory(
        r"D:\Everything\Uni\2022\Sem2\COMP3710\report\AD_NC\test",
        image_size=(240, 256),
        label_mode=None,
        color_mode = "grayscale"
    )

    # Scale to (0, 1)
    train_ds = train_ds.map(scale)
    valid_ds = valid_ds.map(scale)

    train_ds = train_ds.map(
    lambda x: (resize_input(x), x)
    )
    train_ds = train_ds.prefetch(buffer_size=32)

    valid_ds = valid_ds.map(
        lambda x: (resize_input(x), x)
    )
    valid_ds = valid_ds.prefetch(buffer_size=32)


    return train_ds, valid_ds



def scale(image):
    image = image / 255.0
    return image 

def resize_input(input):
    return tf.image.resize(input, [256//4, 240//4], method="area")


        