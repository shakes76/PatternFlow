import sys
from image_histogram import ImageHistogram 

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

	#TODO: get input variables
	#TODO: create ImageHistogram instance	
	#TODO: get output and plot maybe?




if __name__ == '__main__':
	main(sys.argv[1:])
