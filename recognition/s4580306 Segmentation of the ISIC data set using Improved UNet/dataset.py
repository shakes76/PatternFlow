import tensorflow as tf


HEIGHT = 256
WIDTH = 256

training_labels_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Training_Part1_GroundTruth"
test_labels_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Test_v2_Part1_GroundTruth"
validation_labels_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Validation_Part1_GroundTruth"
training_input_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Training_Data"
validation_input_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Validation_Data"
test_input_dir = "C:/Users/Jacob/Downloads/ISIC-2017_Test_v2_Data"


def preprocess_data(filenames):
    data = tf.io.read_file(filenames)

    image = tf.io.decode_jpeg(data, channels=3)

    image = tf.image.resize(image, [HEIGHT, WIDTH])

    image = image / 255.0

    return image


def preprocess_labels(filenames):
    data = tf.io.read_file(filenames)

    image = tf.io.decode_png(data, channels=1)

    image = tf.image.resize(image, [HEIGHT, WIDTH])

    image = image / 255.0

    image = tf.where(image > 0.5, 1.0, 0.0)

    return image


def data_loader(size=256, batch_size=16):
    # input data folders

    # size = 128
    # batch_size = 16
    training_data = tf.data.Dataset.list_files(training_input_dir, shuffle=False)
    training_data = training_data.map(preprocess_data)
    test_data = tf.data.Dataset.list_files(test_input_dir, shuffle=False)
    test_data = test_data.map(preprocess_data)
    validation_data = tf.data.Dataset.list_files(validation_input_dir, shuffle=False)
    validation_data = validation_data.map(preprocess_data)

    training_labels = tf.data.Dataset.list_files(training_labels_dir, shuffle=False)
    training_labels = training_labels.map(preprocess_labels)
    test_labels = tf.data.Dataset.list_files(test_labels_dir, shuffle=False)
    test_labels = test_labels.map(preprocess_labels)
    validation_labels = tf.data.Dataset.list_files(validation_labels_dir, shuffle=False)
    validation_labels = validation_labels.map(preprocess_labels)

    # train_ds = tf.data.Dataset.from_tensor_slices((training_data, training_labels))
    # test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    # validation_ds = tf.data.Dataset.from_tensor_slices((validation_data, validation_labels))
    train_ds = tf.data.Dataset.zip((training_data, training_labels))
    test_ds = tf.data.Dataset.zip((test_data, test_labels))
    validation_ds = tf.data.Dataset.zip((validation_data, validation_labels))
    return train_ds, test_ds, validation_ds

