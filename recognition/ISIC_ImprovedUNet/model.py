import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

ConvProp = dict(kernal_regularizer = l2(0.0005),
                bias_regularizer = l2(0.0005),
				padding = 'same')

def context_module(layer, no_of_filters):
	layer_1 = tf.keras.layers.Conv2D(no_of_filters, (3, 3), **ConvProp)(layer)
	layer_2 = tf.keras.layers.LeakyReLU(alpha = 0.01)(layer_1)
	layer_3 = tf.keras.layers.Dropout(0.3)(layer_2)
	layer_4 = tf.keras.layers.Conv2D(no_of_filters, (3, 3), **ConvProp)(layer_3)
	layer_5 =  tf.keras.layers.LeakyReLU(alpha = 0.01)(layer_4)
	
	return layer_5

def unet():
	input = keras.layers.Input(shape=(256,256,1))

	# Left side of U-Net
	first_layer_conv = tf.keras.layers.Conv2D(16, (3, 3), padding = 'same')(input)
	first_layer_conv = tf.keras.layers.LeakyReLU(alpha = 0.01)(first_layer_conv)
	first_layer_context = context_module(first_layer_conv, 16)
	first_layer = tf.keras.layers.Add()[first_layer_conv, first_layer_context]

	return unet