import tensorflow as tf

test_directory = 'test'
train_directory = 'train'

def load_test():
    data = tf.keras.utils.image_dataset_from_directory(test_directory,image_size=(256, 256),color_mode='grayscale', label_mode='categorical')
    norm = tf.keras.layers.Rescaling(1./255)
    norm_data = data.map(lambda x,y:(norm(x),y))
    return norm_data


def load_train_data():
    train_data = tf.keras.utils.image_dataset_from_directory(train_directory,image_size=(256, 256),color_mode='grayscale', label_mode='categorical',seed = 12, validation_split=0.25, subset='training')
    valid_data = tf.keras.utils.image_dataset_from_directory(train_directory,image_size=(256, 256),color_mode='grayscale', label_mode='categorical',seed = 12, validation_split=0.25, subset='validation')
    norm = tf.keras.layers.Rescaling(1./255)
    train_data = train_data.map(lambda x,y:(norm(x),y))
    valid_data = valid_data.map(lambda x,y:(norm(x),y))
    return train_data, valid_data