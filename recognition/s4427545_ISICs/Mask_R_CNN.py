import tensorflow as tf

class MaskRCNN():

    def __init__(self):
        print('init')

    def train(self):
        print("TF Version:", tf.__version__)
        print(tf.config.list_physical_devices('GPU'))