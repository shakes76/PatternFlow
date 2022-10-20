import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm


BATCH_SIZE = 64
DATA_DIR = 'data/keras_png_slices_data'
TRAIN_DATA = DATA_DIR + '/keras_png_slices_train'
TEST_DATA = DATA_DIR + '/keras_png_slices_test'
VALIDATE_DATA = DATA_DIR + '/keras_png_slices_validate'

def reader(f):
    return tf.io.decode_png(tf.io.read_file(f), channels=1)

def load(files, use_multiprocessing=False):
    if use_multiprocessing:
        import multiprocessing
        pool = multiprocessing.Pool(use_multiprocessing)
        lst = pool.map(reader, tqdm(files))
    else:
        lst = map(reader, tqdm(files))
    
    imgs = np.asarray(list(lst), dtype='float32')
    return imgs

"""
    Load data from predefined paths TRAIN_DATA, TEST_DATA, VALIDATE_DATA.
    optional argument use_multiprocessing defaults to false can specify and integer to spawn child
    processes to load faster on machines with sufficient capabilities
"""
def get_data(use_multiprocessing=False):
    files_train = [os.path.join(TRAIN_DATA, f) for f in os.listdir(TRAIN_DATA) if os.path.isfile(os.path.join(TRAIN_DATA, f))]
    files_test = [os.path.join(TEST_DATA, f) for f in os.listdir(TEST_DATA) if os.path.isfile(os.path.join(TEST_DATA, f))]
    files_validate = [os.path.join(VALIDATE_DATA, f) for f in os.listdir(VALIDATE_DATA) if os.path.isfile(os.path.join(VALIDATE_DATA, f))]

    print("Loading data")
    x_train = load(files_train, use_multiprocessing)
    x_test = load(files_test, use_multiprocessing)
    x_validate = load(files_validate, use_multiprocessing)

    # scale image data to [-1, 1] range
    x_train = x_train/127.5 - 1.0
    x_test = x_test/127.5 - 1.0
    x_validate = x_validate/127.5 - 1.0

    return x_train, x_test, x_validate
