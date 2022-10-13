"""
Loading & preprocessing dataset for COMP3710 VQVAE project
"""

from	glob				import	glob
from	matplotlib.pyplot	import	axis,	imshow, show, subplot
from	numpy				import	array,	reshape
from	PIL					import	Image
import	tensorflow			as		tf

NMALISE			= 255
CENTRE			= 0.5
IMAGE_DIM		= 80
ENC_IN_SHAPE	= (80, 80, 1)

GREYSCALE		= "Greys_r"

SPLITS			= 3
BASE			= "keras_png_slices_data/keras_png_slices_"
TRAINING		= BASE + "train/*"
TESTING			= BASE + "test/*"
VALIDATION		= BASE + "validate/*"

def get_ttv():
	"""
	Read in the training/testing/validation datasets from local files.
	Mostly repurposed from demo 2

	return	- the training, testing and validation datasets
	"""

	#image	= lambda f: (tf.io.decode_png(tf.io.read_file(f)) / NMALISE) - CENTRE
	#to_dset	= lambda d: tf.data.Dataset.from_tensor_slices(d).map(image)
	#npify	= lambda d: array(list(to_dset(d)))

	image	= lambda i: Image.open(i).resize((IMAGE_DIM, IMAGE_DIM))
	npify	= lambda d: array([reshape(image(i), ENC_IN_SHAPE) for i in d])
	normal	= lambda d: (npify(d) / NMALISE) - CENTRE

	train	= glob(TRAINING)
	testing	= glob(TESTING)
	valid	= glob(VALIDATION)

	"""
	srcs	= (glob(TRAINING), glob(TESTING), glob(VALIDATION))
	dsets	=([[]] * SPLITS)

	for i, src in enumerate(srcs):
		for path in src:
			base	= Image.open(path)
			scale	= base.resize((IMAGE_DIM, IMAGE_DIM))
			npify	= reshape(scale, ENC_IN_SHAPE)
			dsets[i].append(npify)

	as_arrs		= tuple([array(d) for d in dsets])
	train_dset, test_dset, val_dset = as_arrs
	norm_train	= (train_dset / NMALISE) - CENTRE

	"""
	"""
	Unsure why I even had this in demo 2 tbh ==> probs unneccesary
	train.sort()
	testing.sort()
	valid.sort()
	"""
	train_dset	= normal(train)
	test_dset	= normal(testing)
	val_dset	= normal(valid)

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
