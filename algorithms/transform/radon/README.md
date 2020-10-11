# Radon Transform
A port of the radon transform from scikit image (skimage.transform.radon()) into tensorflow. This implementation works with both eager execution and computational graphs.

## Algorithm Description
This function takes an image as a tensor and calculates its radon transform (also known as sinogram) for the provided angles. The transformed data for each angle represents a projection along an angle of the input image. This is often used for image processing and reconstruction, for example edge detection.

![Example execution using test.png](https://raw.githubusercontent.com/tm70/PatternFlow/topic-algorithms/transform/radon/example_execution.png)

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
image = imread("test.png", as_gray = True)
image = tf.constant(image)
transformed = radon.radon(image, angles = list(range(20, 60)), circle = True)
transformed = transformed.eval(session=sess)

plt.imshow(transformed)
plt.show()
```

## Notes
While this implementation works with eager execution both on and off, it is highly recommended to run it with eager execution on, as building the computational graph requires an ungodly amount of memory. Further, it runs about 10x slower.

This implementation ports all of the original code into python using tensorflow. The original implementation used Cython to compile backend code to C. As the major cost of this algorithm is doing hundreds of thousands of memory accesses, this significantly improves performance. With eager execution on, this implementation takes about 5000x longer than the original