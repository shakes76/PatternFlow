import tensorflow as tf

# Assumes Directory has a train and test folder. However, functions are written
# to load the data in if there is a defined validation folder already.

# Seed used for generating the training and validation dataset
seed = 123


def normalisation_data(ds):
    """
    Standardises input data and rescales it to be between 0 and 1.

    params:
    ds - The data loader (tf.data.Dataset object) to be normalised
    """
    normalisation_layer = tf.keras.layers.Normalization()
    scaling_layer = tf.keras.layers.Rescaling(1. / 255)
    norm_data = ds.map(lambda x: (normalisation_layer(x)))
    clean_data = norm_data.map(lambda x: (scaling_layer(x)))
    return clean_data


def load_train_data_no_val(path, height, width, batch_size):
    """
    Takes in a directory path, img height, width and batch size and returns normalised training data.
    Use function if no validation data needs to be generated from the given directory.

    params:
    path - Directory path
    height - Height of the images being loaded in
    width - Width of the images being loaded in
    batch_size - Number of images inside a given batch
    """
    train_data = tf.keras.utils.image_dataset_from_directory(path,
                                                             color_mode="rgb",
                                                             labels=None,
                                                             image_size=(height, width),
                                                             batch_size=batch_size)

    return normalisation_data(train_data)


def load_val_from_dir(path, height, width, batch_size):
    """
    Takes in a directory path, img height, width and batch size and returns normalised validation data.
    Use function if validation data is already given in a seperate directory

    params:
    path - Directory path
    height - Height of the images being loaded in
    width - Width of the images being loaded in
    batch_size - Number of images inside a given batch
    """
    val_data = tf.keras.utils.image_dataset_from_directory(path,
                                                           color_mode="rgb",
                                                           labels=None,
                                                           image_size=(height, width),
                                                           batch_size=batch_size)

    return normalisation_data(val_data)


def load_train_data(path, height, width, batch_size, val_split):
    """
    Takes in a directory path, img height, width and batch size and returns normalised training data.
    Use this function for datasets that just have a train and test dataset, and a
    validation dataset needs to be generated.

    Val_split is the proportion of the data to be allocated to validation data.

    params:
    path - Directory path
    height - Height of the images being loaded in
    width - Width of the images being loaded in
    batch_size - Number of images inside a given batch
    val_split - Proportion of the training data to be left out for validation
    """

    train_data = tf.keras.utils.image_dataset_from_directory(path, validation_split=val_split,
                                                             subset="training",
                                                             seed=seed,
                                                             color_mode="rgb",
                                                             labels=None,
                                                             image_size=(height, width),
                                                             batch_size=batch_size)

    return normalisation_data(train_data)


def load_validation_data(path, height, width, batch_size, val_split):
    """
    Takes in a directory path, img height, width and batch size and returns normalised validation data.
    Use function if dataset just has train and test dataset and a validation set needs to be generated

    Val_split is the proportion of the data to be allocated to validation data.

    params:
    path - Directory path
    height - Height of the images being loaded in
    width - Width of the images being loaded in
    batch_size - Number of images inside a given batch
    val_split - Proportion of the training data to be left out for validation
    """
    validation_data = tf.keras.utils.image_dataset_from_directory(path, validation_split=val_split,
                                                                  subset="validation",
                                                                  seed=seed,
                                                                  color_mode="rgb",
                                                                  labels=None,
                                                                  image_size=(height, width),
                                                                  batch_size=batch_size)

    return normalisation_data(validation_data)


def load_test_data(path, height, width, batch_size):
    """
    Takes in a directory path, img height, width and batch size and returns normalised test data.
    Current step up to return 90% of the test data (To reduce the size of test data)

    params:
    path - Directory path
    height - Height of the images being loaded in
    width - Width of the images being loaded in
    batch_size - Number of images inside a given batch
    """
    test_data = tf.keras.utils.image_dataset_from_directory(path, validation_split=0.1,
                                                            subset="training",
                                                            seed=seed,
                                                            color_mode="rgb",
                                                            labels=None,
                                                            image_size=(height, width),
                                                            batch_size=batch_size)

    return normalisation_data(test_data)
