import tensorflow as tf
import os
from tensorflow import keras

class ImageSaver(keras.callbacks.Callback):
    """
    callback that saves output images each epoch
    """
    def __init__(self, relative_filepath, image_count=3):
        """
        callback initialiser
        param relative_filepath: relative filepath
        param image_count: number of images to save each epoch
        """
        self.image_count= image_count
        dirname = os.path.dirname(__file__)
        self.filepath= os.path.join(dirname, relative_filepath)

    def on_epoch_end(self, epoch, logs):
        """ 
        image saving function
        param epoch: current epoch
        """
        generator_inputs = self.model.get_generator_inputs()
        output_images = self.model.generator(generator_inputs)
        output_images *= 256
        output_images.numpy()
        for i in range(self.image_count):
            img = keras.preprocessing.image.array_to_img(output_images[i])
            img.save("{}\epoch_{}_image_{}.png".format(self.filepath, epoch, i))

class WeightSaver(keras.callbacks.Callback):
    """
    callback that saves discriminator and generator model weights each epoch
    """
    def __init__(self, relative_filepath):
        """
        callback initialiser
        param relative_filepath: relative filepath
        """
        dirname = os.path.dirname(__file__)
        self.filepath= os.path.join(dirname, relative_filepath)

    def on_epoch_end(self, epoch, logs):
        """ 
        weight saving function
        param epoch: current epoch
        """
        self.model.generator.save_weights("{}\epoch_{}_generator.h5".format(self.filepath, epoch))
        self.model.discriminator.save_weights("{}\epoch_{}_discriminator.h5".format(self.filepath, epoch))