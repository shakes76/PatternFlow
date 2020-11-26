import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add, DepthwiseConv2D, GlobalAveragePooling2D, Reshape, Multiply, BatchNormalization, Activation, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.utils import normalize

"""
Hyperparameter values for expansion/squeeze ratios, kernel sizes, strides, input/output shapes for the 7-block architecture used in EfficientNet
(https://arxiv.org/abs/1905.11946)
"""

class BlockArgs:
	def __init__(self, kernel_size, num_repeat, input_filters, output_filters, expand_ratio, id_skip, strides, se_ratio):
		self.kernel_size = kernel_size
		self.num_repeat = num_repeat
		self.input_filters = input_filters
		self.output_filters = output_filters
		self.expand_ratio = expand_ratio
		self.id_skip = id_skip
		self.strides = strides
		self.se_ratio = se_ratio

		"""
		Swish activation to make the sigmoid function linear
X : Input tensor
Returns: transformed tensor 
		"""
		
def swish_activation(x):
	return x*K.sigmoid(x)

"""
Normalisation of input values
X : input tensor
Returns : transformed tensor"""
"""

def normalise(tensor):
	X = normalize(tensor)
	return X

"""
Single convolution block for channel wise expansion and feature augmentation before shaping back to the original tensor
"""

def mobile_conv_block(input, block_args, index, classes, final_block=False, axis=3):
#parameter extraction
	kernel_size = block_args.kernel_size
	num_repeat = block_args.num_repeat
	input_filters = block_args.input_filters if (num_repeat == 0) else block_args.output_filters
	first_input_filters = block_args.input_filters
	output_filters = block_args.output_filters
	expand_ratio = block_args.expand_ratio
	id_skip = block_args.id_skip
	strides = block_args.strides if (num_repeat == 0) else (1, 1)
	first_strides = block_args.strides
	se_ratio = block_args.se_ratio
	has_se_ratio = 0 < se_ratio <= 1

#Expansion phase, checks if the same block needs repetition over the network
	first = True
	block_count = 0
	for _ in range(num_repeat):
#expansion phase
		expanded_filters = input_filters*expand_ratio if first else first_input_filters*expand_ratio
		strides_v = first_strides if first else strides
		input_f = first_input_filters if first else input_filters
		X_first = input if first else X
		if first:
			first = False
		X = Conv2D(expanded_filters, 1, padding="same", use_bias=False, name='block'+str(index)+'expansion-'+str(block_count))(X_first)
		X = BatchNormalization(axis=axis)(X)
		X = Activation(swish_activation, name='block'+str(index)+'swish-'+str(block_count))(X)
#Depth wise convolution for separating RGB channels and augmmenting respective feature spaces
		X = DepthwiseConv2D(kernel_size, strides_v, padding="same", use_bias=False, name='block'+str(index)+'depth-wise-'+str(block_count))(X)
		X = BatchNormalization(axis=axis)(X)
		X = Activation(swish_activation)(X)
		
		#Squeeze phase, merging the RGB channels back
		squeezed = GlobalAveragePooling2D(name='block'+str(index)+'global-averaging-'+str(block_count))(X)
		block_count += 1
		squeezed = Reshape((1, 1, expanded_filters))(squeezed)
		squeeze_filters = max(1, int(input_f*se_ratio))
		squeezed = Conv2D(squeeze_filters, 1, activation=swish_activation, padding="same")(squeezed)
		squeezed = Conv2D(expanded_filters, 1, activation="sigmoid", padding="same", use_bias=True)(squeezed)
		X = Multiply()([X, squeezed])
		X = Conv2D(output_filters, 1, padding="same", use_bias=False)(X)
		X = BatchNormalization(axis=axis)(X)
		if id_skip and all(s==1 for s in strides) and input_filters == output_filters:
			X = Dropout(0.2)(X)
	if final_block:
#Checks if all blocks have been added, for the classification phase
		X = Dense(classes, activation='softmax')(X)
		X = GlobalAveragePooling2D()(X)
	return X
