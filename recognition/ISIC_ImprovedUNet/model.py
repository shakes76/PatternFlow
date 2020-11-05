import tensorflow as tf
from tensorflow import keras

ConvProp = dict(padding = 'same')

def context_module(layer, no_of_filters):
	layer_1 = tf.keras.layers.Conv2D(no_of_filters, (3, 3), **ConvProp)(layer)
	layer_2 = tf.keras.layers.LeakyReLU(alpha = 0.01)(layer_1)
	layer_3 = tf.keras.layers.Dropout(0.3)(layer_2)
	layer_4 = tf.keras.layers.Conv2D(no_of_filters, (3, 3), **ConvProp)(layer_3)
	layer_5 =  tf.keras.layers.LeakyReLU(alpha = 0.01)(layer_4)
	
	return layer_5

def upsampling_module(layer, no_of_filters):
	layer_1 = tf.keras.layers.UpSampling2D(size = (2, 2))(layer)
	layer_2 = tf.keras.layers.Conv2D(no_of_filters, (3, 3), **ConvProp)(layer_1)
	layer_3 = tf.keras.layers.LeakyReLU(alpha = 0.01)(layer_2)

	return layer_3

def localisation_module(layer, no_of_filters):
	layer_1 = tf.keras.layers.Conv2D(no_of_filters, (3, 3), **ConvProp)(layer)
	layer_2 = tf.keras.layers.LeakyReLU(alpha = 0.01)(layer_1)
	layer_3 = tf.keras.layers.Conv2D(no_of_filters, (1, 1), **ConvProp)(layer_2)
	layer_4 = tf.keras.layers.LeakyReLU(alpha = 0.01)(layer_3)

	return layer_4

def unet():
	start = keras.layers.Input(shape=(256,256,1))

	# Left side of UNet
	first_layer_conv = tf.keras.layers.Conv2D(16, (3, 3), padding = 'same')(input)
	first_layer_conv = tf.keras.layers.LeakyReLU(alpha = 0.01)(first_layer_conv)
	first_layer_context = context_module(first_layer_conv, 16)
	first_layer = tf.keras.layers.Add()[first_layer_conv, first_layer_context]

	second_layer_conv = tf.keras.layers.Conv2D(32, (3, 3), strides = 2, **ConvProp)(first_layer)
	second_layer_conv = tf.keras.layers.LeakyReLU(alpha = 0.01)(second_layer_conv)
	second_layer_context = context_module(second_layer_conv, 32)
	second_layer = tf.keras.layers.Add()[second_layer_conv, second_layer_context]

	third_layer_conv = tf.keras.layers.Conv2D(64, (3, 3), strides = 2, **ConvProp)(second_layer)
	third_layer_conv = tf.keras.layers.LeakyReLU(alpha = 0.01) (third_layer_conv)
	third_layer_context = context_module(third_layer_conv, 64)
	third_layer = tf.keras.layers.Add()[third_layer_conv, third_layer_context]

	fourth_layer_conv = tf.keras.layers.Conv2D(128, (3, 3), strides = 2, **ConvProp)(third_layer)
	fourth_layer_conv = tf.keras.layers.LeakyReLU(alpha = 0.01)(fourth_layer_conv)
	fourth_layer_context = context_module(fourth_layer_conv, 128)
	fourth_layer = tf.keras.layers.Add()[fourth_layer_conv, fourth_layer_context]

	# Bottom of UNet
	fifth_layer_conv = tf.keras.layers.Conv2D(256, (3, 3), strides = 2, **ConvProp)(fourth_layer)
	fifth_layer_conv = tf.keras.layers.LeakyReLU(alpha = 0.01)(fifth_layer_conv)
	fifth_layer_context = context_module(fifth_layer_conv, 256)
	fifth_layer = tf.keras.layers.Add()[fifth_layer_conv, fifth_layer_context]
	fifth_layer_up = upsampling_module(fifth_layer, 128)

	# Right side of UNet
	sixth_layer = tf.concat([fourth_layer, fifth_layer_up], axis = 3)
	sixth_layer_local = localisation_module(sixth_layer, 128)
	sixth_layer_up = upsampling_module(sixth_layer_local, 64)

	seventh_layer = tf.concat([third_layer, sixth_layer_up], axis = 3)
	seventh_layer_local = localisation_module(seventh_layer, 64)
	seventh_layer_up = upsampling_module(seventh_layer_local, 32)

	eighth_layer = tf.concat([second_layer, seventh_layer_up], axis = 3)
	eighth_layer_local = localisation_module(eighth_layer, 32)
	eighth_layer_up = upsampling_module(eighth_layer_local, 16)

	ninth_layer = tf.concat([first_layer, eighth_layer_up], axis = 3)
	ninth_layer_conv = tf.keras.layers.Conv2D(32, (3, 3), **ConvProp)(ninth_layer)
	ninth_layer_conv = tf.keras.layers.LeakyReLU(alpha = 0.01)(ninth_layer_conv)

	# Segmentation Layer
	seg_first_layer_conv = tf.keras.layers.Conv2D(32, (3, 3), **ConvProp)(seventh_layer_up)
	seg_first_layer_conv = tf.keras.layers.LeakyReLU(alpha = 0.01)(seg_first_layer_conv)
	seg_first_layer_up = upsampling_module(seg_first_layer_conv, 32)

	seg_second_layer_conv = tf.keras.layers.Conv2D(32, (3, 3), **ConvProp)(eighth_layer_up)
	seg_second_layer_conv = tf.keras.layers.LeakyReLU(alpha = 0.01)(seg_second_layer_conv)
	seg_second_layer = tf.keras.layers.Add()[seg_first_layer_up, seg_second_layer_conv]
	seg_second_layer_up = upsampling_module(seg_second_layer, 32)

	seg_third_layer_conv = tf.keras.layers.Conv2D(32, (3, 3), **ConvProp)(ninth_layer_conv)
	seg_third_layer_conv = tf.keras.layers.LeakyReLU(alpha = 0.01)(seg_third_layer_conv)
	seg_third_layer = tf.keras.layers.Add()[seg_second_layer_up, seg_third_layer_conv]

	# Output
	finish = tf.keras.layers.Conv2D(1, (1, 1), activation = 'softmax')(seg_third_layer)
	unet = keras.Model(inputs = start, output = finish)

	return unet