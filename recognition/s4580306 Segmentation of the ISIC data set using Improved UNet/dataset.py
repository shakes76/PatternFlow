import tensorflow as tf
import glob
import pathlib

# input data folders
training_labels_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Training_Data"
test_labels_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Test_Part1_GroundTruth"
validation_labels_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Validation_Part1_GroundTruth"

size = 256
batch_size = 16
# training_input_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Training_Data"

# training_input_dir = tf.keras.utils.get_file(origin=training_input_url)
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
                                                            validation_split=0.3,
                                                            subset="training")
validation_data = tf.keras.utils.image_dataset_from_directory(training_input_dir,
                                                              labels=None,
                                                              batch_size=batch_size,
                                                              image_size=(size, size),
                                                              validation_split=0.15,
                                                              subset="validation")
validation_data = tf.keras.utils.image_dataset_from_directory(training_input_dir,
                                                              labels=None,
                                                              batch_size=batch_size,
                                                              image_size=(size, size),
                                                              validation_split=0.15,
                                                              subset="validation")