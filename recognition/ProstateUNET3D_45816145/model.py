from tensorflow import keras
from tensorflow.keras import layers

def unet3d(filters, data_dimensions, class_count):
	# Construct keras model
	first_layer = layers.Input((*data_dimensions, 1))

	# We will pass through "previous" to represent last layer
	previous = first_layer

	# LEFT SIDE
	downscale_layers = len(filters)
	downscale_filters = filters
	downscale_tails = []
	for i in range(downscale_layers):
		previous = layers.Conv3D(downscale_filters[i], (3, 3, 3), padding='same', activation='relu')(previous)
		previous = layers.Conv3D(downscale_filters[i], (3, 3, 3), padding='same', activation='relu')(previous)
		
		if i != downscale_layers - 1:
			downscale_tails.append(previous)
			previous = layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2))(previous)
			previous = layers.Dropout(0.2)(previous)
			
	# RIGHT SIDE
	# reverse references, since upscale is looped other way
	downscale_tails = list(reversed(downscale_tails))

	upscale_layers = len(filters) - 1
	upscale_filters = list(reversed(filters))[1:]
	for i in range(upscale_layers):
		# Up Convolution
		previous = layers.Conv3DTranspose(upscale_filters[i], (2, 2, 2), strides=(2, 2, 2))(previous)
		previous = layers.Dropout(0.2)(previous)
		
		# Pull across
		tail = downscale_tails[i]
		previous = layers.concatenate([previous, tail])
		
		# Convolutions
		previous = layers.Conv3D(upscale_filters[i], (3, 3, 3), padding='same', activation='relu')(previous)
		previous = layers.Conv3D(upscale_filters[i], (3, 3, 3), padding='same', activation='relu')(previous)
			
	last_layer = layers.Conv3D(class_count, (1, 1, 1), activation='softmax')(previous)

	return keras.Model(first_layer, last_layer)