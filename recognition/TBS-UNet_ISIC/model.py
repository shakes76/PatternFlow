import tensorflow as tf
from tensorflow.keras import datasets, layers, models

input_layer = layers.Input((256, 192, 3))


"""
### STANDARD U-NET
# down path

# down-level 1
input_layer = layers.Input((256,192,3))
conv_1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(input_layer)
conv_2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv_1)
mPool_1 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(conv_2)

# down-level 2
conv_3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(mPool_1)
conv_4 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(conv_3)
mPool_2 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(conv_4)

# down-level 3
conv_5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(mPool_2)
conv_6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv_5)
mPool_3 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(conv_6)

# down-level 4
conv_7 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(mPool_3)
conv_8 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv_7)
mPool_4 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(conv_8)

# bottom-level
conv_9 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(mPool_4)
conv_10 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv_9)

# up path
# up-level 4
uconv_1 = layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(conv_10)
cat_1 = layers.Concatenate()([uconv_1,conv_8])
conv_11 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(cat_1)
conv_12 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv_11)

# up-level 3
uconv_2 = layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv_12)
cat_2 = layers.Concatenate()([uconv_2, conv_6])
conv_13 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(cat_2)
conv_14 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv_13)

# up-level 2
uconv_3 = layers.Conv2DTranspose(128, (2, 2), strides=(2,2), padding='same')(conv_14)
cat_3 = layers.Concatenate()([uconv_3, conv_4])
conv_15 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(cat_3)
conv_16 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv_15)

# up-level 1
uconv_4 = layers.Conv2DTranspose(64, (2, 2), strides=(2,2), padding='same')(conv_16)
cat_4 = layers.Concatenate()([uconv_4, conv_2])
conv_17 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(cat_4)
conv_18 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv_17)
conv_19 = layers.Conv2D(4, (1, 1), activation='softmax', padding='same')(conv_18)

unet = tf.keras.Model(inputs=input_layer, outputs=conv_19)
unet.compile(optimizer='adam', loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
print(unet.summary())"""