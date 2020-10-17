import glob
import unittest

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from recognition.s4436194_oasis_dcgan.data_helper import Dataset
from recognition.s4436194_oasis_dcgan.oasis_dcgan import (
    DCGANModelFramework,
    DATA_TRAIN_DIR,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
)
from recognition.s4436194_oasis_dcgan.models_helper import make_generator_model_basic, make_discriminator_model


def run_dcgan_training():

    batch_size = 8
    epochs = 3

    framework = DCGANModelFramework()
    framework.train_dcgan(batch_size=batch_size, epochs=epochs)


def run_dcgan_tests():
    framework = DCGANModelFramework()
    framework.test_dcgan()


class DriverTests(unittest.TestCase):
    """
    Use this framework to test individual parts of the library
    """

    def test_get_batches(self):
        """
        Test the get batches method

        Get batches yields images as numpy arrays that are then used as training batches. We can get
        one of these batches and check the images are correct/the output has the correct format
        """

        test_batch_size = 16
        dataset = Dataset(glob.glob(f"{DATA_TRAIN_DIR}/*.png"), IMAGE_WIDTH, IMAGE_HEIGHT)
        batch = next(dataset.get_batches(batch_size=test_batch_size))

        # Print the batch using matplotlib
        fig = plt.figure(figsize=(4, 4))

        for i in range(batch.shape[0]):
            plt.subplot(4, 4, i + 1)

            image = batch[i, :, :, 0]
            image = (((image - image.min()) * 255) / (image.max() - image.min())).astype(np.uint8)
            plt.imshow(image, cmap="Greys")
            plt.axis('off')

        plt.show()

        self.assertIsInstance(batch, np.ndarray)
        self.assertEquals(batch.shape, (test_batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, 1))

        print("Done")

    def test_generator_model(self):
        """
        Output of the generator should be a 256 x 256 image
        """
        model = make_generator_model_basic(100)
        input_ = tf.random.normal([1, 100])
        output = model(input_)

        # Check the output is correct
        self.assertIsInstance(output, tf.Tensor)
        self.assertEquals(output.shape, tf.TensorShape((1, 256, 256, 1)))

    def test_discriminator_model(self):
        """
        Output of the generator should be a single value tensor
        """

        model = make_discriminator_model(256, 256)
        input_ = tf.random.normal([1, 256, 256, 1])
        output = model(input_)

        # Check the output is correct
        self.assertIsInstance(output, tf.Tensor)
        self.assertEquals(output.shape, tf.TensorShape((1, 1)))


if __name__ == '__main__':
    # run_dcgan_training()
    # run_dcgan_tests()

    unittest.main()
