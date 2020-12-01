from Histogram import histogram
from skimage import data
from skimage import data, img_as_float
import matplotlib.pyplot as plt

"""
        COMP3710 Open source project
        Name: Chrisbin James
        Student Number: 45431780
        
        Test driver script for Tensorflow port of skimage.exposure.histogram
"""

# import RGB image with integer color values
image = data.chelsea()
plt.imshow(image)
plt.title('Sample Integer-RGB image')
plt.show()

# red channel
red = image[:,:,0]
# green channel
green = image[:,:,1]
# blue channel
blue = image[:,:,2]

channel = ['Red', 'Green', 'Blue']
fig = plt.figure(figsize=(30,10))
for i in range(3):
    ax = fig.add_subplot(1, 3, i+1)
    histograms, bins = histogram(image=image[:,:,i])
    ax.plot(bins, histograms)
    ax.set_title(channel[i]+' channel')
    ax.set_xlabel('Pixel Count')
    ax.set_ylabel('Color Value')

# import a grayscale image with float color values
image_float = img_as_float(data.camera())
plt.imshow(image_float)
plt.title('Sample Float-Grayscale image')
plt.show()
# plot histogram for the image
fig = plt.figure(figsize=(30,10))
histograms, bins = histogram(image=image_float)
plt.plot(bins, histograms,)
plt.title('Histogram for Float-Grayscale image')
plt.xlabel('Colour Value')
plt.ylabel('Pixel count')
plt.show()
