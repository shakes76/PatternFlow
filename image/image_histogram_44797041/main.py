import sys
from image_histogram import ImageHistogram 
from matplotlib import pyplot as plt

# only to open the image
from PIL import Image

"""
Driver script for ImageHistogram module
"""


def main(args):
	"""
	"""
	fileName = args[0]
	img = Image.open(fileName)
	hist = ImageHistogram(img)
	plt.plot(hist.hist)
	plt.show()




if __name__ == '__main__':
	main(sys.argv[1:])
