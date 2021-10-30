"""Data Pre-processing
"""
import tensorflow as tf
import glob
import os

# ISIC training input folder name
ISIC_INPUT_DIR = "ISIC2018_Task1-2_Training_Input"
# ISIC training ground truth folder name
ISIC_GROUNDTRUTH_DIR = "ISIC2018_Task1_Training_GroundTruth"


def get_filenames(isic_dir):
    """Get the filenames of the images in the ISIC dataset.

    Args:
        isic_dir (string): The directory of the ISIC dataset.

    Returns:
        features (list): The list of the filenames of the images in the ISIC dataset.
        labels (list): The list of the filenames of the labels of the images in the ISIC dataset.
    """
    feature_dir = os.path.join(
        isic_dir, ISIC_INPUT_DIR, '*.jpg')

    # the dataset actually shuffles here due to arbitrary reading order
    features = glob.glob(feature_dir)
    # make sure the features and labels have the same order
    labels = [f.replace(ISIC_INPUT_DIR, ISIC_GROUNDTRUTH_DIR).replace(
        '.jpg', '_segmentation.png') for f in features]

    print("Number of images loaded:", len(features), len(labels))
    return features, labels


def split_data(features, labels, validation_split=0.2, test_split=0.2):
    """Split the data into training, validation and test sets.

    Args:
        features (list): The list of the filenames of the images in the ISIC dataset.
        labels (list): The list of the filenames of the labels of the images in the ISIC dataset.
        validation_split (float, optional): The proportion of the data to use for validation. Defaults to 0.2.
        test_split (float, optional): The proportion of the data to use for testing. Defaults to 0.2.

    Returns:
        train_features (list): The list of the filenames of the training images in the ISIC dataset.
        train_labels (list): The list of the filenames of the training labels of the images in the ISIC dataset.
        val_features (list): The list of the filenames of the validation images in the ISIC dataset.
        val_labels (list): The list of the filenames of the validation labels of the images in the ISIC dataset.
        test_features (list): The list of the filenames of the testing images in the ISIC dataset.
        test_labels (list): The list of the filenames of the testing labels of the images in the ISIC dataset.
    """
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
    """Load the image file and resize it to the given size.

    Args:
        image_file (string): The filename of the image.
        image_size (list): The size of the image (height, width).

    Returns:
        image (tensor): The image tensor.
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)

    # normalise the image
    image = tf.cast(image, tf.float32) / 255.0
    return image


def __load_labels(label_file, image_size, num_classes):
    """Load the label file and resize it to the given size.

    Args:
        label_file (string): The filename of the label.
        image_size (list): The size of the image (height, width).
        num_classes (int): The number of classes.

    Returns:
        label (tensor): The label tensor.
    """
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
    """Create a dataset from the given features and labels.

    Args:
        features (list): The list of the filenames of the images in the ISIC dataset.
        labels (list): The list of the filenames of the labels of the images in the ISIC dataset.
        image_size (list): The size of the image (height, width).
        num_classes (int): The number of classes.

    Returns:
        dataset (tf.data.Dataset): The created dataset.
    """
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
    """Create the training, validation and testing datasets.

    Args:
        train_featrues (list): The list of the filenames of the training images in the ISIC dataset.
        train_labels (list): The list of the filenames of the training labels of the images in the ISIC dataset.
        val_features (list): The list of the filenames of the validation images in the ISIC dataset.
        val_labels (list): The list of the filenames of the validation labels of the images in the ISIC dataset.
        test_features (list): The list of the filenames of the testing images in the ISIC dataset.
        test_labels (list): The list of the filenames of the testing labels of the images in the ISIC dataset.
        image_size (list): The size of the image (height, width).
        num_classes (int): The number of classes.

    Returns:
        train_dataset (tf.data.Dataset): The training dataset.
        val_dataset (tf.data.Dataset): The validation dataset.
        test_dataset (tf.data.Dataset): The testing dataset.
    """
    train_set = __create_dataset(train_featrues, train_labels,
                                 image_size, num_classes)
    val_set = __create_dataset(val_features, val_labels,
                               image_size, num_classes)
    test_set = __create_dataset(test_features, test_labels,
                                image_size, num_classes)
    return train_set, val_set, test_set
