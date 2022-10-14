import os
import sys
import zipfile

import requests
import tensorflow as tf

dataset_location = "https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA/download"
dataset_directory = "dataset/"
dataset_zip_name = "keras_png_slices_data.zip"
dataset_folder_name = "keras_png_slices_data/"
dataset_train_folder = "keras_png_slices_train"
dataset_test_folder = "keras_png_slices_test"
dataset_val_folder = "keras_png_slices_validate"

batch_size = 32

# Download the dataset, if it hasn't already been downloaded
def download_dataset():
    # Create the dataset directory
    if not os.path.isdir(dataset_directory):
        os.mkdir(dataset_directory)

    # Download dataset zip
    print(f"Downloading dataset into ./{dataset_directory}{dataset_zip_name}")
    if not os.path.exists(dataset_directory + dataset_zip_name):
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
    else:
        print("Dataset already downloaded.\n")

# Unzip the dataset
def unzip_dataset():
    print(f"Extracting dataset into ./{dataset_directory}{dataset_folder_name}")
    if not os.path.isdir(dataset_directory + dataset_folder_name):
        with zipfile.ZipFile(dataset_directory + dataset_zip_name) as z:
            z.extractall(path=dataset_directory)
        print("Dataset extracted.\n")
    else:
        print("Dataset already extracted.\n")

# Load the dataset
def load_dataset() -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
    print("Loading training data...")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_directory + dataset_folder_name + dataset_train_folder,
        labels=None,
        batch_size=batch_size,
    )
    print("Loading testing data...")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_directory + dataset_folder_name + dataset_test_folder,
        labels=None,
        batch_size=batch_size,
    )
    print("Loading validation data...")
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_directory + dataset_folder_name + dataset_val_folder,
        labels=None,
        batch_size=batch_size,
    )
    return (train_ds, test_ds, val_ds)

# Preprocess the data
def preprocess_data(
    train_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset
) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
    # Normalize the data around 0.0
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x: (normalization_layer(x) - 0.5))
    test_ds = test_ds.map(lambda x: (normalization_layer(x) - 0.5))
    val_ds = val_ds.map(lambda x: (normalization_layer(x) - 0.5))

    return (train_ds, test_ds, val_ds)

# Load and preprocess the dataset
def get_dataset() -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
    download_dataset()
    unzip_dataset()
    train_ds, test_ds, val_ds = load_dataset()
    train_ds, test_ds, val_ds = preprocess_data(train_ds, test_ds, val_ds)

    return (train_ds, test_ds, val_ds)

if __name__ == "__main__":
    get_dataset()
