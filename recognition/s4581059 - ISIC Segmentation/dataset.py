from sklearn.model_selection import train_test_split
from glob import glob
import tensorflow as tf
#Contains the data loader for loading and preprocessing data


img_height = 64
img_width = 64
batch_size = 32

def load_data(path):
    """
    Loads a ISIC data set - where each image is scaled to 64x64 to assist with 
    performance/load, dataset is loaded in batches of 32. With a validation split
    of 0.2

    Returns: Raw datasets in the form: training_dataset, validation_dataset
    """

    training_dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    return training_dataset, validation_dataset

if __name__ == "__main__":
    print(load_data("C:/Users/danie/Downloads/ISIC DATA/"))