import glob
import os
import tensorflow as tf 

def process_data(path):
    train_dir = sorted(glob.glob(path + "\keras_png_slices_train\*.png"))
    test_dir = sorted(glob.glob(path + "\keras_png_slices_test\*.png"))
    
    train = tf.data.Dataset.from_tensor_slices(train_dir)
    test = tf.data.Dataset.from_tensor_slices(test_dir)

    train = train.shuffle(len(train_dir))
    test = test.shuffle(len(test_dir))

    train_images = train.take(50)
    real_images = tf.convert_to_tensor(list(train_images.as_numpy_iterator()))

    return train.map(process_path).batch(16), test.map(process_path).batch(16), real_images

def process_path(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels = 1)
    img = tf.image.resize(img, (256,256))
    img = tf.cast(img, tf.float32) - 127.5
    img = img / 127.5

    return img
    