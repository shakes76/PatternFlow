import tensorflow as tf

class ImageHistogram:
	"""
	Implements a histogram of a given image.
	"""

	def __init__(self, image, nbins=256, source_range = 'image', normalize = False):
		self.hist, self.bin_centres = self.create_hist(image, nbins, source_range, normalize)
	

	def create_hist(self,image, nbins, source_range, normalize):
		"""
		"""
		sess = tf.InteractiveSession()		
		image = tf.constant(image)

		source_range = tf.constant([0,256], dtype = tf.float64)
		shape = tf.shape(image)

		if (tf.size(shape) == 3):
			print("If this is a colour image, the histogram will be computed on the flattened image.\
				You can instead apply this function to each color channel.")

		image_flatten = tf.reshape(image, [-1])
		image_flatten = tf.dtypes.cast(image_flatten, tf.float64)
		
		hist = tf.histogram_fixed_width(image_flatten, source_range, nbins=nbins)
		bins = tf.histogram_fixed_width_bins(image_flatten, source_range, nbins=nbins)

		bin_centres = (bins[:-1] + bins[1:]) / 2
		

		if (normalize):
			hist = hist / tf.reduce_sum(hist)
		
		return hist.eval(), bin_centres.eval()


