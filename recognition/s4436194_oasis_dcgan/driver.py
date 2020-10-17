import glob
import unittest

import matplotlib.pyplot as plt
import numpy as np

from recognition.s4436194_oasis_dcgan.data_helper import Dataset
from recognition.s4436194_oasis_dcgan.oasis_dcgan import (
    DCGANModelFramework,
    DATA_TRAIN_DIR,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
)


def run_dcgan_training():
    framework = DCGANModelFramework()
    framework.train_dcgan(batch_size=32, epochs=3)


def run_dcgan_tests():
    framework = DCGANModelFramework()
    framework.test_dcgan()


class DriverTests(unittest.TestCase):
    """
    Use this framework to test individual parts of the library
    """

    def test_get_batches(self):
        """Test the get batches method"""

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


if __name__ == '__main__':
    run_dcgan_training()
    # run_dcgan_tests()

    unittest.main()
