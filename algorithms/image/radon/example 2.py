# Author: E-Hern Lee
# License: CC0
# Copyright: (c) 2019, E-Hern Lee
# Created: 18/10/2019
# Modified: 07/11/2019
# Description: Displays canonical example for Radon transform, the Shepp-Logan phantom, and its Radon transform using source and ported algorithms

import tensorflow as tf
import matplotlib.pyplot as plt
import radon

image = tf.image.decode_png(tf.io.read_file('test/phantom.png'))
result = radon.radon(image)
plt.figure(figsize=(10, 8))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original image')
plt.subplot(1, 3, 2)
plt.imshow(result[:, :, 0], cmap='gray')
plt.title('Radon (Ours)')
plt.subplot(1, 3, 3)
# this one is a single channel png, which matplotlib can't handle directly
plt.imshow(tf.image.decode_png(tf.io.read_file('test/phantom_radon.png'))[:, :, 0], cmap='gray')
plt.title('Radon (scikit-learn)')
plt.show()
