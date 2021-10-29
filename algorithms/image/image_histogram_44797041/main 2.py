import sys
from image_histogram import ImageHistogram 
# only to plot
from matplotlib import pyplot as plt

# only to read image
from PIL import Image


"""
Driver script for ImageHistogram module

Usage:

python3 main.py <image>
OR python main.py <image>

e.g python3 main.py coins.png


"""


def main(args):
	"""
	Main function for example use of ImageHistogram (get the histogram of an image 
	in tensorflow)
	"""
	#open image
	fileName = args[0]
	img = Image.open(fileName)
	
	#use implemented algorithm to get histogram and plot
	hist = ImageHistogram(img, nbins = 256, source_range = 'dtype')
	fig, axs = plt.subplots(1,2, figsize = (15,7))
	axs[0].set_title('Original Image')
	axs[0].imshow(img)
	axs[1].set_title('Image histogram values')
	axs[1].plot(hist.hist)
	plt.show()
	fig.savefig('example_output.png')




if __name__ == '__main__':
	main(sys.argv[1:])
