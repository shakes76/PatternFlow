# Wiener Deconvolution
Deconvolution is a method of restoring data processed by convolution. 
It is widely used in signal processing and image processing. 
For example, deconvolution is significantly effective for image denoising. 
Wiener Deconvolution processes data by applying the Wiener filter, 
the Wiener filter minimizes the mean square error between the estimated random process and the desired process.
## Installation
##### Dependencies:
- Python 3.6
- Tensorflow 1.10

Use Git to clone the repository,
```sh
git clone https://github.com/microwen/PatternFlow.git
```
Or [Download](https://github.com/microwen/PatternFlow/archive/topic-algorithms.zip).

Add the module directory to syspath.
```python
from sys import path
path.append("PATH/TO/Wiener_deconv")
```
## Example
Here's an example of how to use wiener deconvolution.
```python
from wiener import wiener
import numpy as np
psf = np.ones((5, 5)) / 25 # Point Spread Function
img_denoised = wiener(img_noise, psf, 2) # Apply wiener deconvolution to 'img_noise'
```
Another example for 'camera' from [scikit-image](https://scikit-image.org/docs/dev/api/skimage.data.html)
```sh
python PATH/TO/wiener_deconv/main.py
```
