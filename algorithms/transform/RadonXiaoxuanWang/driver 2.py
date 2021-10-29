import numpy as np
import matplotlib.pyplot as plt
from skimage.data import checkerboard

image = checkerboard()

# display the original image
plt.imshow(image)

from radon_transform import radon

# apply the radon transform
theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta)

# show the transformed image
plt.imshow(sinogram)

# apply the inverse radon transform to confirm the correctness
from skimage.transform import iradon

reconstruction = iradon(sinogram, theta=theta, circle=True)

plt.imshow(reconstruction)
