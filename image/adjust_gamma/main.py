from skimage import data, img_as_float
from adjust_gamma import adjust_gamma

image = img_as_float(data.chelsea())

adjusted = adjust_gamma(image)