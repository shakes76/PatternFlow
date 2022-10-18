import tensorflow as tf
import glob
import pathlib


def normalise_data(data):
    data = tf.cast(data, tf.float32) / 255.0
    return data


def normalise_label(data):
    data = tf.cast(data, tf.float32) / 255.0
    data = tf.math.round(data)
    return data


def data_loader(size=128, batch_size=16):
    # input data folders
    training_labels_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Training_Part1_GroundTruth"
    test_labels_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Test_v2_Part1_GroundTruth"
    validation_labels_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Validation_Part1_GroundTruth"
    training_input_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Training_Data"
    validation_input_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Validation_Data"
    test_input_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Test_v2_Data"

    # size = 128
    # batch_size = 16

    training_labels = tf.keras.utils.image_dataset_from_directory(training_labels_dir,
                                                                  labels=None,
                                                                  color_mode='grayscale',
                                                                  batch_size=batch_size,
                                                                  image_size=(size, size))
    test_labels = tf.keras.utils.image_dataset_from_directory(test_labels_dir,
                                                              labels=None,
                                                              color_mode='grayscale',
                                                              batch_size=batch_size,
                                                              image_size=(size, size))
    validation_labels = tf.keras.utils.image_dataset_from_directory(validation_labels_dir,
                                                                    labels=None,
                                                                    color_mode='grayscale',
                                                                    batch_size=batch_size,
                                                                    image_size=(size, size))
    training_data = tf.keras.utils.image_dataset_from_directory(training_input_dir,
                                                                labels=None,
                                                                batch_size=batch_size,
                                                                image_size=(size, size),
                                                                )
    validation_data = tf.keras.utils.image_dataset_from_directory(validation_input_dir,
                                                                  labels=None,
                                                                  batch_size=batch_size,
                                                                  image_size=(size, size)
                                                                  )
    test_data = tf.keras.utils.image_dataset_from_directory(test_input_dir,
                                                            labels=None,
                                                            batch_size=batch_size,
                                                            image_size=(size, size)
                                                            )
    training_data = training_data.map(normalise_data)
    test_data = test_data.map(normalise_data)
    validation_data = validation_data.map(normalise_data)
    training_labels = training_labels.map(normalise_label)
    test_labels = test_labels.map(normalise_label)
    validation_labels = validation_labels.map(normalise_label)

    # train_ds = tf.data.Dataset.from_tensor_slices((training_data, training_labels))
    # test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    # validation_ds = tf.data.Dataset.from_tensor_slices((validation_data, validation_labels))
    train_ds = tf.data.Dataset.zip((training_data, training_labels))
    test_ds = tf.data.Dataset.zip((test_data, test_labels))
    validation_ds = tf.data.Dataset.zip((validation_data, validation_labels))
    return train_ds, test_ds, validation_ds

