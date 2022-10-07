from pkgutil import get_data
import numpy as np
import tensorflow as tf
import tensorflow.keras.utils

     


"""
Retrieves the datasets from the folder for use by the algorithm

@return the three datasets for training, validation, and testing

"""
def get_datasets(train_path, test_path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path, color_mode = "grayscale", 
        labels = "inferred", 
        label_mode = "binary",
        validation_split = 0.1,
        seed = 123,
        subset = "training")
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_path, color_mode = "grayscale", 
        labels = "inferred", 
        label_mode = "binary",
        validation_split = 0.1,
        seed = 123,
        subset = "validation")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path, color_mode = "grayscale", 
        labels = "inferred", 
        label_mode = "binary")

    return train_ds, val_ds, test_ds


def main():
    train_path = "dataset\\train"
    test_path = "dataset\\test"

    train_ds, val_ds, test_ds = get_datasets(train_path, test_path)

if __name__ == "__main__":
    main()