## Components for the network

import tensorflow as tf

# Classes
class path:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test

img_path = path(
    '/Users/wotah_man/Documents/UQ/ISIC Dataset/Train',
    '/Users/wotah_man/Documents/UQ/ISIC Dataset/Val',
    '/Users/wotah_man/Documents/UQ/ISIC Dataset/Test'
)

# Functions
def normalize(dataset):
    '''normalize the image data to 0~1 float'''
    for img, lbl in dataset:
        x = tf.math.divide(img, 255.0)
        y = lbl
    return x, y