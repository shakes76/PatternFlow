import matplotlib
import matplotlib.pyplot as plt
from skimage import data
from equalize_hist import equalize_hist

matplotlib.rcParams['font.size'] = 9

# Load an example image
gray_img = data.moon()
colour_img = data.astronaut()

# Apply histogram equalisation
gray_img_eq = equalize_hist(gray_img)
colour_img_eq = equalize_hist(colour_img)

# Display results
fig = plt.figure()

axl = fig.add_subplot(2, 2, 1)
axl.imshow(gray_img)
axl = fig.add_subplot(2, 2, 2)
axl.imshow(gray_img_eq)
axl = fig.add_subplot(2, 2, 3)
axl.imshow(colour_img)
axl = fig.add_subplot(2, 2, 4)
axl.imshow(colour_img_eq)

fig.tight_layout()
plt.show()