import matplotlib.pyplot as plt
from skimage import data, img_as_float
from adjust_gamma import adjust_gamma

image = img_as_float(data.chelsea())

adjusted = adjust_gamma(image)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Source')

plt.subplot(1, 2, 2)
plt.imshow(adjusted)
plt.title('Adjusted')

plt.tight_layout()
plt.show()