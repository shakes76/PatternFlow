import tensorflow as tf

test_directory = 'test'
train_directory = 'train'

def load_test():
    return tf.keras.utils.image_dataset_from_directory(test_directory)


def load_train_data():
    train_data = tf.keras.utils.image_dataset_from_directory(train_directory,image_size=(256, 240),color_mode='grayscale',seed = 12, validation_split=0.25, subset='training')
    valid_data = tf.keras.utils.image_dataset_from_directory(train_directory,image_size=(256, 240),color_mode='grayscale',seed = 12, validation_split=0.25, subset='validation')
    return train_data, valid_data