import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from equalize_hist import equalize_hist

matplotlib.rcParams['font.size'] = 9

# Load example images
gray_img = data.moon()
colour_img = data.astronaut()

# Apply histogram equalisation
gray_img_eq = equalize_hist(gray_img)
colour_img_eq = equalize_hist(colour_img)

# Calculate histograms to show difference
gray_img_hist = np.histogram(gray_img.flatten(), bins=256)
gray_img_eq_hist = np.histogram(gray_img_eq.flatten(), bins=256)
colour_img_hist = np.histogram(colour_img.flatten(), bins=256)
colour_img_eq_hist = np.histogram(colour_img_eq.flatten(), bins=256)

# Display results
fig = plt.figure()

axl = fig.add_subplot(4, 2, 1)
axl.imshow(gray_img)
axl.title.set_text('original moon')

axl = fig.add_subplot(4, 2, 2)
axl.plot(gray_img_hist[0])

axl = fig.add_subplot(4, 2, 3)
axl.imshow(gray_img_eq)
axl.title.set_text('equalised moon')

axl = fig.add_subplot(4, 2, 4)
axl.plot(gray_img_eq_hist[0])

axl = fig.add_subplot(4, 2, 5)
axl.imshow(colour_img)
axl.title.set_text('original astronaut')

axl = fig.add_subplot(4, 2, 6)
axl.plot(colour_img_hist[0])

axl = fig.add_subplot(4, 2, 7)
axl.imshow(colour_img_eq)
axl.title.set_text('equalised astronaut')

axl = fig.add_subplot(4, 2, 8)
axl.plot(colour_img_eq_hist[0])

fig.tight_layout()
plt.savefig('hist.png', bbox_inches='tight')
plt.show()