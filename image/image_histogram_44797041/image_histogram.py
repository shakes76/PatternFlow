import tensorflow as tf
"""
"""

class ImageHistogram:
	"""
	"""

	def __init__(self, image, nbins=256, source_range = 'image', normalize = False):
		sess = tf.InteractiveSession()		
		image = tf.constant(image)
		shape = tf.shape(image)
		if (tf.size(shape) == 3):
			print("If this is a colour image, the histogram will be computed on the flattened image.\
				You can instead apply this function to each color channel.")

		image_flatten = tf.reshape(image, [-1])
		print(shape.eval())







