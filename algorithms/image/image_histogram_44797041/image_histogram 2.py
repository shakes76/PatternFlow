import tensorflow as tf

class ImageHistogram:
	"""
	Implements a histogram of a given image.
	"""

	def __init__(self, image, nbins=256, source_range = 'image', normalize = False):
		self.hist, self.bin_centres = self.create_hist(image, nbins, source_range, normalize)
	

	def create_hist(self,image, nbins, source_range, normalize):
		"""
		Creates a histogram of a given image. The histogram is computed on the flattened image,
		so for colour images, the function should be used separately on each channel to obtain a histogram 
		for each colour channel.

		Parameters
		__________
		
		image: array - an array representation of the image
		nbins: int (optional) - number of bins used to calculate histogram
		source_range: string (optional) - 'image' (default) gets the range from the input image,
        'dtype' determines the range from the expected range of the images of that data type.
		normalize: bool (optional) - If True, the histogram will be normalized by the sum of its values.

		Returns
		_______
		
		hist: array - the values of the histogram
		bin_centers: array - the values of center of the bins.

		Example
		_______

		See main.py for example script

		"""
		# check the shape of image
		shape = tf.shape(image)

		if (tf.size(shape) == 3):
			print("If this is a colour image, the histogram will be computed on the flattened image.\
				You can instead apply this function to each color channel.")


		# setup
		sess = tf.InteractiveSession()		
		image = tf.constant(image)
		
		# flatten image
		image_flatten = tf.reshape(image, [-1])

		# specify the source range
		if (source_range == 'image'):

			# general range
			min_val = tf.reduce_min(image_flatten).eval()
			max_val = tf.reduce_max(image_flatten).eval()
			hist_range = tf.constant([min_val, max_val], dtype = tf.float64)

		elif (source_range == 'dtype'):
			
			# get the limits of the type
			hist_range = tf.DType(image_flatten.dtype).limits
			hist_range = tf.constant(hist_range, dtype = tf.float64)

		else:
			print('Wrong value for `source range` parameter')

		# cast 
		image_flatten = tf.dtypes.cast(image_flatten, tf.float64)
		
		# get values and bin edges of the histogram	
		hist = tf.histogram_fixed_width(image_flatten, hist_range, nbins=nbins)
		bins = tf.histogram_fixed_width_bins(image_flatten, hist_range, nbins=nbins)

		bin_centres = (bins[:-1] + bins[1:]) / 2
		
		# normalize if specified
		if (normalize):
			hist = hist / tf.reduce_sum(hist)
		
		return hist.eval(), bin_centres.eval()


