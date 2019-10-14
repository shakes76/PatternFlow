from skimage import data
import matplotlib.pyplot as plt
from equalize_hist import equalize_hist

img = data.moon()
img_eq = equalize_hist(img)

fig = plt.figure()
axl = fig.add_subplot(1, 2, 1)
axl.imshow(img)
axl.title.set_text('original')
axl = fig.add_subplot(1, 2, 2)
axl.imshow(img_eq)
axl.title.set_text('equalised')

plt.show()
plt.savefig('simple.png', bbox_inches='tight')