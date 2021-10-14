import glob
import tensorflow as tf
import os

data_directory = "/home/tomdx/datasets/akoa_dataset"

data_directory_glob = glob.glob(data_directory)

batch_size = 1
img_height = 228
img_width = 260

def get_data() -> tf.data.Dataset:

    # Get labels
    labels = []
    for root_name, dir_names, file_names in os.walk(data_directory):
        file_names.sort()
        for file_name in file_names:
            file_name = file_name.lower().replace("_", "")
            if "left" in file_name:
                labels.append(0)
            elif "right" in file_name:
                labels.append(1)

    return tf.keras.utils.image_dataset_from_directory(
        "/home/tomdx/datasets/",
        labels=labels,
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

