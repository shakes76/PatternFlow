import tensorflow as tf
import os

def load_paths(directory):
    training_paths = []
    if not os.path.isfile(training_binary_path):
        for filename in os.listdir(directory):  
            path = os.path.join(directory,filename)
            training_paths.append(path)
            
        random.shuffle(training_paths)
        return training_paths


def load_images(filename):
    img = tf.io.read_file(filename)
    img = tf.io.decode_png(img, channels = 1)
    img = tf.image.resize(img, size = (256, 256))
    img = img / 255.0
    return img
