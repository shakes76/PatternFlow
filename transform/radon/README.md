# Radon Transform
A port of the radon transform from scikit image (skimage.transform.radon()) into tensorflow. This implementation works with both eager execution and computational graphs.

## Algorithm Description
TODO

## Dependencies
The implementation itself (radon.py) requires:
* tensorflow (built using version 2.0.0)
* Python math module

The test script (test.py) additionally requires:
* scikit-image
* matplotlib
* Python time module
* Python sys module

The additional dependencies are only for visualisation and comparison.

## Example use
radon() takes 3 arguments:
* The image to be transformed (as a tensor)
* A list of angles for which the transform should be calculated (Optional, defaults to list(range(180))
* Whether or not to assume the image is 0 outside the circle (Optional, defaults to True)

Example use with eager execution off:
```
import radon
import tensorflow as tf
from skimage.io import imread   # for loading the image
import matplotlib.pyplot as plt # for visualising the image and its transform

sess = tf.compat.v1.Session()
image = imread("image.png", as_gray = True)
image = tf.constant(image)
transformed = radon.radon(image, angles = list(range(60)), circle = True)
transformed = transformed.eval(session=sess)

plt.imshow(transformed)
plt.show()
```