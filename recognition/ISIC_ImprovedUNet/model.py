"""
Author: Marko Uksanovic
Student Number: s4484509
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

# Global constants
ConvProp = dict(padding = 'same', 
                kernel_initializer='he_uniform',
				kernel_regularizer=regularizers.l2(0.0001))
leakyReLU = tf.keras.layers.LeakyReLU(alpha = 0.01)

def unet():
	"""
	The Improved UNet model, based on that of https://arxiv.org/pdf/1802.10508v1.pdf

	Returns:
		The trained UNet
	"""
	start = keras.Input(shape=(256,256,1))

	# Left Side of UNet
	first_layer_conv = tf.keras.layers.Conv2D(16, (3, 3), activation = leakyReLU, **ConvProp)(start)
	first_layer_conv2 = tf.keras.layers.Conv2D(16, (3, 3), activation = leakyReLU, **ConvProp)(first_layer_conv)
	first_layer_conv2 = tf.keras.layers.Dropout(0.3)(first_layer_conv2)
	first_layer_conv2 = tf.keras.layers.Conv2D(16, (3, 3), activation = leakyReLU, **ConvProp)(first_layer_conv2)
	first_layer = tf.keras.layers.Add()([first_layer_conv, first_layer_conv2])

	second_layer_conv = tf.keras.layers.Conv2D(32, (3, 3), strides = (2, 2), activation = leakyReLU, **ConvProp)(first_layer)
	second_layer_conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation = leakyReLU, **ConvProp)(second_layer_conv)
	second_layer_conv2 = tf.keras.layers.Dropout(0.3)(second_layer_conv2)
	second_layer_conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation = leakyReLU, **ConvProp)(second_layer_conv2)
	second_layer = tf.keras.layers.Add()([second_layer_conv, second_layer_conv2])

	third_layer_conv = tf.keras.layers.Conv2D(64, (3, 3), strides = (2, 2), activation = leakyReLU, **ConvProp)(second_layer)
	third_layer_conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation = leakyReLU, **ConvProp)(third_layer_conv)
	third_layer_conv2 = tf.keras.layers.Dropout(0.3)(third_layer_conv2)
	third_layer_conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation = leakyReLU, **ConvProp)(third_layer_conv2)
	third_layer = tf.keras.layers.Add()([third_layer_conv, third_layer_conv2])

	fourth_layer_conv = tf.keras.layers.Conv2D(128, (3, 3), strides = (2, 2), activation = leakyReLU, **ConvProp)(third_layer)
	fourth_layer_conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation = leakyReLU, **ConvProp)(fourth_layer_conv)
	fourth_layer_conv2 = tf.keras.layers.Dropout(0.3)(fourth_layer_conv2)
	fourth_layer_conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation = leakyReLU, **ConvProp)(fourth_layer_conv2)
	fourth_layer = tf.keras.layers.Add()([fourth_layer_conv, fourth_layer_conv2])

	# Bottom of UNet
	fifth_layer_conv = tf.keras.layers.Conv2D(256, (3, 3), strides = (2, 2), activation = leakyReLU, **ConvProp)(fourth_layer)
	fifth_layer_conv2 = tf.keras.layers.Conv2D(256, (3, 3), activation = leakyReLU, **ConvProp)(fifth_layer_conv)
	fifth_layer_conv2 = tf.keras.layers.Dropout(0.3)(fifth_layer_conv2)
	fifth_layer_conv2 = tf.keras.layers.Conv2D(256, (3, 3), activation = leakyReLU, **ConvProp)(fifth_layer_conv2)
	fifth_layer = tf.keras.layers.Add()([fifth_layer_conv, fifth_layer_conv2])

	fifth_layer_up = tf.keras.layers.UpSampling2D(size = (2, 2))(fifth_layer)
	fifth_layer_up = tf.keras.layers.Conv2D(128, (2, 2), activation = leakyReLU, **ConvProp)(fifth_layer_up)
	fifth_layer_up = tf.concat([fourth_layer, fifth_layer_up], axis = 3)

	# Right side of UNet
	sixth_layer_local = tf.keras.layers.Conv2D(128, (3, 3), activation = leakyReLU, **ConvProp)(fifth_layer_up)
	sixth_layer_local = tf.keras.layers.Conv2D(128, (1, 1), activation = leakyReLU, **ConvProp)(sixth_layer_local)
	sixth_layer_up = tf.keras.layers.UpSampling2D(size = (2, 2))(sixth_layer_local)
	sixth_layer_up = tf.keras.layers.Conv2D(64, (2, 2), activation = leakyReLU, **ConvProp)(sixth_layer_up)
	sixth_layer = tf.concat([third_layer, sixth_layer_up], axis = 3)

	seventh_layer_local = tf.keras.layers.Conv2D(64, (3, 3), activation = leakyReLU, **ConvProp)(sixth_layer)
	seventh_layer_local = tf.keras.layers.Conv2D(64, (1, 1), activation = leakyReLU, **ConvProp)(seventh_layer_local)
	seventh_layer_up = tf.keras.layers.UpSampling2D(size = (2, 2))(seventh_layer_local)
	seventh_layer_up = tf.keras.layers.Conv2D(32, (2, 2), activation = leakyReLU, **ConvProp)(seventh_layer_up)
	seventh_layer = tf.concat([second_layer, seventh_layer_up], axis = 3)

	eighth_layer_local = tf.keras.layers.Conv2D(32, (3, 3), activation = leakyReLU, **ConvProp)(seventh_layer)
	eighth_layer_local = tf.keras.layers.Conv2D(32, (1, 1), activation = leakyReLU, **ConvProp)(eighth_layer_local)
	eighth_layer_up = tf.keras.layers.UpSampling2D(size = (2, 2))(eighth_layer_local)
	eighth_layer_up = tf.keras.layers.Conv2D(16, (2, 2), activation = leakyReLU, **ConvProp)(eighth_layer_up)
	eighth_layer = tf.concat([first_layer, eighth_layer_up], axis = 3)

	# Segmentation Layer
	seg_first_layer = tf.keras.layers.Conv2D(32, (3, 3), activation = leakyReLU, **ConvProp)(eighth_layer)
	seg_first_layer = tf.keras.layers.Conv2D(2, (1, 1), activation = leakyReLU, **ConvProp)(seg_first_layer)

	seg_second_layer = tf.keras.layers.Conv2D(2, (1, 1), activation = leakyReLU, **ConvProp)(eighth_layer_local)

	seg_third_layer = tf.keras.layers.Conv2D(2, (1, 1), activation = leakyReLU, **ConvProp)(seventh_layer_local)

	seg_third_layer_up = tf.keras.layers.UpSampling2D(size = (2, 2))(seg_third_layer)
	seg_second_layer = tf.keras.layers.Add()([seg_third_layer_up, seg_second_layer])
	seg_second_layer = tf.keras.layers.UpSampling2D(size = (2, 2))(seg_second_layer)
	seg_out = tf.keras.layers.Add()([seg_first_layer, seg_second_layer])

	# Output
	finish = tf.keras.layers.Conv2D(2, (1, 1), activation = 'softmax')(seg_out)
	unet = keras.Model(inputs = start, outputs = finish)

	return unet