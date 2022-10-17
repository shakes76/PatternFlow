import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 128
DATA_DIR = 'data/keras_png_slices_data'
TRAIN_DATA = DATA_DIR + '/keras_png_slices_train'
TEST_DATA = DATA_DIR + '/keras_png_slices_test'
VALIDATE_DATA = DATA_DIR + '/keras_png_slices_validate'

def reader(f):
    return tf.io.decode_png(tf.io.read_file(f), channels=1)

def load(files):
    lst = map(reader, tqdm(files))
    imgs = np.asarray(list(lst), dtype='float32')
    return imgs


def get_data():
    files_train = [os.path.join(TRAIN_DATA, f) for f in os.listdir(TRAIN_DATA) if os.path.isfile(os.path.join(TRAIN_DATA, f))]
    files_test = [os.path.join(TEST_DATA, f) for f in os.listdir(TEST_DATA) if os.path.isfile(os.path.join(TEST_DATA, f))]
    files_validate = [os.path.join(VALIDATE_DATA, f) for f in os.listdir(VALIDATE_DATA) if os.path.isfile(os.path.join(VALIDATE_DATA, f))]

    print("Loading data")
    x_train = load(files_train)
    x_test = load(files_test)
    x_validate = load(files_validate)

    mean = np.mean(x_train)
    var = np.mean(x_train)

    x_train = x_train/255.0 - 0.5
    x_test = x_test/255.0 - 0.5
    x_validate = x_validate/255.0 - 0.5

    return x_train, x_test, x_validate, mean, var
