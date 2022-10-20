import tensorflow as tf
from tensorflow import keras
import os
import shutil


def setup_folders():
    # Create directory paths.
    valid_directory = "valid-split"
    train_directory = "train-split"
    parent_dir = os.getcwd()

    valid_path = os.path.join(parent_dir, valid_directory)
    train_path = os.path.join(parent_dir, train_directory)

    source_path = os.path.join(parent_dir, "train")

    source_ad_path = os.path.join(source_path, "AD")
    source_nc_path = os.path.join(source_path, "NC")

    valid_ad = os.path.join(valid_path, "AD")
    valid_nc = os.path.join(valid_path, "NC")

    train_ad = os.path.join(train_path, "AD")
    train_nc = os.path.join(train_path, "NC")

    try:
        # If directories already exist will cause exception.
        os.mkdir(valid_path)
        os.mkdir(valid_ad)
        os.mkdir(valid_nc)

        os.mkdir(train_path)
        os.mkdir(train_ad)
        os.mkdir(train_nc)

        ad_files = os.listdir(source_ad_path)

        # Alzheimer's images
        # Split flag is used to ensure 80% of images go into train and 20% go into validation.
        split_flag = 0
        # Previous ID ensures patients with the same ID end up in the same dataset.
        previous_id = 0
        for filename in ad_files:
            if filename.split("_")[0] == previous_id:
                if split_flag == 0:
                    shutil.copy2(os.path.join(source_ad_path, filename), valid_ad)
                else:
                    shutil.copy2(os.path.join(source_ad_path, filename), train_ad)
            else:
                split_flag = (split_flag + 1) % 5
                if split_flag == 0:
                    shutil.copy2(os.path.join(source_ad_path, filename), valid_ad)
                else:
                    shutil.copy2(os.path.join(source_ad_path, filename), train_ad)

                previous_id = filename.split("_")[0]

        nc_files = os.listdir(source_nc_path)

        # Healthy Images
        # Split flag is used to ensure 80% of images go into train and 20% go into validation.
        split_flag = 0
        # Previous ID ensures patients with the same ID end up in the same dataset.
        previous_id = 0
        for filename in nc_files:
            if filename.split("_")[0] == previous_id:
                if split_flag == 0:
                    shutil.copy2(os.path.join(source_nc_path, filename), valid_nc)
                else:
                    shutil.copy2(os.path.join(source_nc_path, filename), train_nc)
            else:
                split_flag = (split_flag + 1) % 5
                if split_flag == 0:
                    shutil.copy2(os.path.join(source_nc_path, filename), valid_nc)
                else:
                    shutil.copy2(os.path.join(source_nc_path, filename), train_nc)

                previous_id = filename.split("_")[0]

    except OSError:
        pass


def get_train_data():
    AUTOTUNE = tf.data.AUTOTUNE
    train_data = keras.utils.image_dataset_from_directory(
        directory='train-split/',
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(256, 240))

    class_names = train_data.class_names
    print(class_names)

    train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)


    return train_data


def get_valid_data():
    AUTOTUNE = tf.data.AUTOTUNE
    valid_data = keras.utils.image_dataset_from_directory(
        directory='valid-split/',
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(256, 240))

    valid_data = valid_data.cache().prefetch(buffer_size=AUTOTUNE)
    return valid_data


def get_test_data():
    AUTOTUNE = tf.data.AUTOTUNE
    test_data = keras.utils.image_dataset_from_directory(
        directory='test/',
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(256, 240))

    test_data = test_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    return test_data

if __name__ == '__main__':
    setup_folders()
    validationData = get_valid_data()

    for data, labels in validationData.take(1):
        print(data.shape)
        print(labels.shape)

