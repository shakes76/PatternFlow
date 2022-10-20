import os
import sys
import zipfile

import requests
import numpy as np
import tensorflow as tf

dataset_location = "https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA/download"
dataset_directory = "dataset/"
dataset_zip_name = "keras_png_slices_data.zip"
dataset_folder_name = "keras_png_slices_data/"
dataset_train_folder = "keras_png_slices_train"
dataset_test_folder = "keras_png_slices_test"
dataset_val_folder = "keras_png_slices_validate"

image_size = (128, 128)

# Download the dataset, if it hasn't already been downloaded
def download_dataset():
    # Create the dataset directory
    if not os.path.isdir(dataset_directory):
        os.mkdir(dataset_directory)

    # Download dataset zip
    if not os.path.exists(dataset_directory + dataset_zip_name):
        print(f"Downloading dataset into ./{dataset_directory}{dataset_zip_name}")
        response = requests.get(dataset_location, stream=True)
        total_length = response.headers.get("content-length")

        with open(dataset_directory + dataset_zip_name, "wb") as f:
            # Show download progress bar (doesn't work on non-Unix systems)
            # Adapted from https://stackoverflow.com/a/15645088
            if total_length is None or not os.name == "posix":
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[8%sD%s]" % ('=' * done, ' ' * (50-done)))
                    sys.stdout.write(" %d%%" % int(dl / total_length * 100))
                    sys.stdout.flush()
            print()

        print("Dataset downloaded.\n")
    #else:
        #print("Dataset already downloaded.\n")

# Unzip the dataset
def unzip_dataset():
    if not os.path.isdir(dataset_directory + dataset_folder_name):
        print(f"Extracting dataset into ./{dataset_directory}{dataset_folder_name}")
        with zipfile.ZipFile(dataset_directory + dataset_zip_name) as z:
            z.extractall(path=dataset_directory)
        print("Dataset extracted.\n")
    #else:
        #print("Dataset already extracted.\n")

# Load the dataset
def load_dataset(folder: str) -> tf.data.Dataset:
    #print("Loading dataset...")
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_directory + dataset_folder_name + folder,
        labels=None,
        image_size=image_size,
    )
    return ds

# Scale the given image to a range of [-0.5, 0.5] and change it to 1 colour channel
def _scale_image(image: tf.Tensor) -> tf.Tensor:
    image = image / 255 - 0.5
    image = tf.image.rgb_to_grayscale(image)
    return image

# Preprocess the data
def preprocess_data(dataset: tf.data.Dataset) -> np.array:
    return np.asarray(list(dataset.unbatch().map(_scale_image)))

# Load and preprocess the training dataset
def get_train_dataset() -> np.array:
    download_dataset()
    unzip_dataset()

    train_ds = load_dataset(dataset_train_folder)
    train_ds = preprocess_data(train_ds)

    return train_ds

# Load and preprocess the testing dataset
def get_test_dataset() -> np.array:
    download_dataset()
    unzip_dataset()

    test_ds = load_dataset(dataset_test_folder)
    test_ds = preprocess_data(test_ds)

    return test_ds

if __name__ == "__main__":
    get_train_dataset()
