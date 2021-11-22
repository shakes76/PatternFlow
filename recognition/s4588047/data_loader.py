import tensorflow as tf

batch_size = 256

def load_data(path):
    '''
    Load the images from the path and returns them in a tuple of tensors of 
    train and validate

            Parameters:
                    path: path string of the dataset

            Returns:
                    x_train, x_validate: a tuple of two data arrays
    '''
    # gets images dataset of each data catagory
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        label_mode=None,
        image_size=(128, 128),
        batch_size=batch_size,
        subset='training',
        validation_split=0.2,
        seed = 123
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        label_mode=None,
        image_size=(128, 128),
        batch_size=batch_size,
        subset='validation',
        validation_split=0.2,
        seed = 123
    )
    return process_data(train_ds, val_ds)


def process_data(train, validate):
    '''
    Takes the train and validate TF Dataset and converts them to two arrays
    and normalises the scale between 0-255.

            Parameters:
                    train: the training TF Dataset object
                    validate: the validation TF Dataset object

            Returns:
                    x_train, x_validate: a tuple of two data arrays 
                    (normalised)
    '''
    x_train = tf.concat([tf.image.rgb_to_grayscale(x) for x in train], axis=0)
    x_validate = tf.concat([tf.image.rgb_to_grayscale(x) for x in validate], 
                           axis=0)
    # scale between 0 to 255.
    x_train = tf.cast(x_train, 'float32') / 255.
    x_validate = tf.cast(x_validate, 'float32') / 255.
    
    return x_train, x_validate
