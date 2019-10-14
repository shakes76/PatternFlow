import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from equalize_hist import equalize_hist

matplotlib.rcParams['font.size'] = 9

# Load an example image
gray_img = data.moon()
colour_img = data.astronaut()

# Create boolean mask for grayscale images
image_mask = np.zeros(gray_img.shape)
image_mask[256:, 0:] = 1

# Apply histogram equalisation
gray_img_eq = equalize_hist(gray_img)
gray_img_eq_mask = equalize_hist(gray_img, mask=image_mask)
colour_img_eq = equalize_hist(colour_img)

# Calculate histograms to show difference
gray_img_hist = np.histogram(gray_img.flatten(), bins=256)
gray_img_eq_hist = np.histogram(gray_img_eq.flatten(), bins=256)

# Display results
fig = plt.figure()

axl = fig.add_subplot(3, 2, 1)
axl.imshow(gray_img)
axl.title.set_text('original')

axl = fig.add_subplot(3, 2, 2)
axl.imshow(gray_img_eq)
axl.title.set_text('gray equalised')

axl = fig.add_subplot(3, 2, 3)
axl.plot(gray_img_hist[0])
axl.title.set_text('original')

axl = fig.add_subplot(3, 2, 4)
axl.plot(gray_img_eq_hist[0])
axl.title.set_text('equalised w/ mask')

axl = fig.add_subplot(3, 2, 5)
axl.imshow(colour_img)
axl.title.set_text('original')

axl = fig.add_subplot(3, 2, 6)
axl.imshow(colour_img_eq)
axl.title.set_text('colour equalised')

fig.tight_layout()
plt.show()