"""
OASIS DCGAN Driver tests

Used for testing the DCGAN model framework, namely the get batches methods and models. This is done in a
UnitTests framework for convenience.

@author nthompson97
"""

import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import unittest

from recognition.s4436194_oasis_dcgan.models_helper import (
    make_models_64
)
from recognition.s4436194_oasis_dcgan.oasis_dcgan import (
    DATA_TRAIN_DIR,
    Dataset
)


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
        image_width = 128
        image_height = 128

        dataset = Dataset(glob.glob(f"{DATA_TRAIN_DIR}/*.png"), image_width, image_height)
        batch = next(dataset.get_batches(batch_size=test_batch_size))

        # Print the batch using matplotlib
        fig = plt.figure(figsize=(4, 4))

        for i in range(batch.shape[0]):
            plt.subplot(4, 4, i + 1)

            image = batch[i, :, :, 0].numpy()
            image = (((image - image.min()) * 255) / (image.max() - image.min())).astype(np.uint8)
            plt.imshow(image, cmap="gray")
            plt.axis('off')

        plt.show()

        self.assertIsInstance(batch, tf.Tensor)
        self.assertEquals(batch.shape, tf.TensorShape((test_batch_size, image_width, image_height, 1)))

        print("Done")

    def test_generator_model(self):
        """
        Output of the generator should be a 256 x 256 image. Only testing an untrained model, so separate to
        the run_dcgan_tests method above
        """

        _, generator, size = make_models_64()
        input_ = tf.random.normal([1, 100])
        output = generator(input_)

        # Check the output is correct
        self.assertIsInstance(output, tf.Tensor)
        self.assertEquals(output.shape, tf.TensorShape((1, size, size, 1)))

    def test_discriminator_model(self):
        """
        Output of the generator should be a single value tensor
        """

        discriminator, _, size = make_models_64()
        input_ = tf.random.normal([1, size, size, 1])
        output = discriminator(input_)

        # Check the output is correct
        self.assertIsInstance(output, tf.Tensor)
        self.assertEquals(output.shape, tf.TensorShape((1, 1)))


if __name__ == '__main__':
    unittest.main()
