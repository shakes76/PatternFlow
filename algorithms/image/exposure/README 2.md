## Scikit-image exposure class reimplementation in pytorch

### Algorithms

Algorithms in exposure module are used in image correction and showing distribution
information. Only four algorithms were implemented from this module.

#### histogram

return the histogram of an image with bins centered.
```python
from skimage import data
import torch
image = torch.tensor(data.camera())
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image)
ax1.set_title("original image")
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
hist, bins = histogram(image)
ax2.set_title("histogram of image")
ax2.hist(hist, bins=bins)
```
![hist](https://i.imgur.com/9KenQHd.png)

#### equlize_hist

return image after histogram equalization

```python
from skimage import data
import torch
image = torch.tensor(data.camera())
equ_image = equalize_hist(image)
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("original image")
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax1.imshow(image)
ax2.set_title("after equalization")
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.imshow(equ_image)
plt.show()
```
![equal](https://i.imgur.com/HJyl3QN.png)

#### cumulative_distribution

return cumulative distribution function (cdf) for the given image
```python
from skimage import data
import torch
image = torch.tensor(data.coffee())
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image)
ax1.set_title("original image")
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
hist, bins = cumulative_distribution(image)
ax2.plot(bins, hist, 'b-')
ax2.set_title("cdf of image")
plt.show()
```

![cdf](https://i.imgur.com/0gzlPqO.png)

#### adjust_gamma

performs gamma correction on the input image. For gamma greater than 1, the
output image will be darker than the original one. However, when gamma less
than 1, the output image will be brighter than the original one.
```python
from skimage import data
import torch
image = torch.tensor(data.astronaut())
shift_left_img = adjust_gamma(image, 1.5)
shift_right_img = adjust_gamma(image, 0.5)
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set_title("original image")
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax1.imshow(image)
ax2.set_title("adjust gamma 1.5")
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.imshow(shift_left_img)
ax3.set_title("adjust gamma 0.5")
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ax3.imshow(shift_right_img)
plt.show()
 ```

 ![gamma](https://i.imgur.com/kWcmbKL.png)

### Dependencies

this project is tested in following context:

-   os: macOS Mojave@10.14
-   python: python@3.6
-   linter: mypy@0.73

Only one external library was used inside algorithm implementation, it is
pytorch@1.2.0. However, there are two external libraries were used in the test,
one is numpy another is skimage.
