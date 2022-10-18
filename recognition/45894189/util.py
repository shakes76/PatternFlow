import tensorflow as tf
from tensorflow import keras

class ImageSaver(keras.callbacks.Callback):
    def __init__(self, image_count=3):
        self.image_count= image_count

    def on_epoch_end(self, epoch, logs):
        z = [tf.random.normal((32, 512)) for i in range(7)]
        noise = [tf.random.uniform([32, res, res, 1]) for res in [4, 8, 16, 32, 64, 128, 256]] #TODO: Global variable these upscaling values somewhere
        input = tf.ones([32, 4, 4, 512])
        output_images = self.model.generator([input, z, noise])
        output_images *= 256
        output_images.numpy()
        for i in range(self.image_count):
            img = keras.preprocessing.image.array_to_img(output_images[i])
            img.save("generated_img_%03d_%d.png" % (epoch, i))

class WeightSaver(keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs):
        pass