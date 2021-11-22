"""
Extracts patches from input array

@author: Pritish Roy
@email: pritish.roy@uq.edu.au
"""

import tensorflow as tf

from settings.config import *


class Patches(tf.keras.layers.Layer):
    """Divides an image into patches taking parameters from settings.

    Patches are created of 40 x 40 with stride length 40. Thus creating
    IMAGE_SIZE // PATCH_SIZE = 228 // 40 = 5 patches in each stride's and a
    total of 25 patches across the image.
    """
    def __init__(self):
        super(Patches, self).__init__()

        self.patch_size = PATCH_SIZE

    def call(self, images):
        """
        Extracting patches from image array as shown in the example below:
            If we mark the pixels in the input image which are taken
            for the output with *, we see the following:

                   *  *  *  4  5  *  *  *  9 10
                   *  *  * 14 15  *  *  * 19 20
                   *  *  * 24 25  *  *  * 29 30
                  31 32 33 34 35 36 37 38 39 40
                  41 42 43 44 45 46 47 48 49 50
                   *  *  * 54 55  *  *  * 59 60
                   *  *  * 64 65  *  *  * 69 70
                   *  *  * 74 75  *  *  * 79 80
                  81 82 83 84 85 86 87 88 89 90
                  91 92 93 94 95 96 97 98 99 100

               # Yields:
                  [[[[ 1  2  3 11 12 13 21 22 23]
                     [ 6  7  8 16 17 18 26 27 28]]
                    [[51 52 53 61 62 63 71 72 73]
                     [56 57 58 66 67 68 76 77 78]]]]
        :param images:
        :return:
        """
        patches = tf.image.extract_patches(images=images,
                                           sizes=[1, self.patch_size,
                                                  self.patch_size, 1],
                                           strides=[1, self.patch_size,
                                                    self.patch_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding=PADDING)
        return tf.reshape(patches, [tf.shape(images)[0], -1, patches.shape[-1]])
