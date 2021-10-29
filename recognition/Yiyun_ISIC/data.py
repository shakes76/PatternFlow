"""Data Pre-processing
"""
import tensorflow as tf
import glob
import os


def get_filenames(isic_dir):
    feature_dir = os.path.join(
        isic_dir, 'ISIC2018_Task1-2_Training_Input', '*.jpg')

    # the dataset actually shuffles here due to arbitrary reading order
    features = glob.glob(feature_dir)
    # make sure the features and labels have the same order
    labels = [f.replace('ISIC2018_Task1-2_Training_Input',
                        'ISIC2018_Task1_Training_GroundTruth').replace('.jpg', '_segmentation.png') for f in features]

    print("Number of images loaded:", len(features), len(labels))
    return features, labels


def split_data(features, labels, validation_split=0.2, test_split=0.2):
    # calculate the split size
    training_split = 1 - (validation_split + test_split)
    num_train = int(training_split * len(features))
    num_val = int(validation_split * len(features))
    num_test = len(features) - num_train - num_val

    print("Number of training images:", num_train)
    print("Number of validation images:", num_val)
    print("Number of test images:", num_test)

    # split the features and labels into training set, validation set and test set
    train_features, val_features, test_features = tf.split(
        features, [num_train, num_val, num_test])
    train_labels, val_labels, test_labels = tf.split(
        labels, [num_train, num_val, num_test])

    return train_features, train_labels, val_features, val_labels, test_features, test_labels


def __load_features(image_file, image_size):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)

    # normalise the image
    image = tf.cast(image, tf.float32) / 255.0
    return image


def __load_labels(label_file, image_size, num_classes):
    label = tf.io.read_file(label_file)
    label = tf.image.decode_png(label, channels=1)
    label = tf.image.resize(label, image_size)

    # convert the label to one-hot encoding
    label = label / 255.0
    label = tf.cast(label, tf.int32)
    label = tf.squeeze(label, axis=2)
    label = tf.one_hot(label, depth=num_classes)
    return label


def __create_dataset(features, labels, image_size, num_classes):
    # create and shuffle dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(len(features))

    # load features and labels
    dataset = dataset.map(
        lambda feature, label: (__load_features(feature, image_size),
                                __load_labels(label, image_size, num_classes)))

    return dataset


def create_datasets(train_featrues, train_labels, val_features, val_labels,
                    test_features, test_labels, image_size, num_classes):
    train_set = __create_dataset(train_featrues, train_labels,
                                 image_size, num_classes)
    val_set = __create_dataset(val_features, val_labels,
                               image_size, num_classes)
    test_set = __create_dataset(test_features, test_labels,
                                image_size, num_classes)
    return train_set, val_set, test_set
