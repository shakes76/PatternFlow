"""
Loading & preprocessing dataset for COMP3710 VQVAE project
"""

from	glob				import	glob
from	matplotlib.pyplot	import	axis, imshow, show, subplot
import	tensorflow			as		tf

NMALISE		= 255
CENTRE		= 0.5

GREYSCALE	= "Greys_r"

TRAINING	= "keras_png_slices_data/keras_png_slices_train/*"
TESTING		= "keras_png_slices_data/keras_png_slices_test/*"
VALIDATION	= "keras_png_slices_data/keras_png_slices_validate/*"

def get_ttv():
	"""
	Read in the training/testing/validation datasets from local files.
	Mostly repurposed from demo 2

	return	- tuple<list<EagerTensor>> the training, testing and v'dation datasets
	"""

	image	= lambda f: (tf.io.decode_png(tf.io.read_file(f)) / NMALISE) - CENTRE
	to_dset	= lambda d: list(tf.data.Dataset.from_tensor_slices(d).map(image))

	train	= glob(TRAINING)
	testing	= glob(TESTING)
	valid	= glob(VALIDATION)
	"""
	Unsure why I even had this in demo 2 tbh ==> probs unneccesary
	train.sort()
	testing.sort()
	valid.sort()
	"""
	train_dset	= to_dset(train)
	test_dset	= to_dset(testing)
	val_dset	= to_dset(valid)

	print(type(train_dset[0]))

	return (train_dset, test_dset, val_dset)

def preview(dataset, n):
	"""
	Show the first n^2 images of the dataset in a n x n grid

	dataset	- training / testing / validation dataset to preview
	n		- length of preview square grid
	"""

	for i in range(n):
		for j in range(n):
			ind = (n * i) + j + 1
			subplot(n, n, ind)
			axis("off")
			imshow(dataset[ind], cmap = GREYSCALE)

	show()
