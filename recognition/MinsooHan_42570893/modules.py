import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Concatenate, UpSampling2D, Input, Activation, add, \
    BatchNormalization, Dropout, Softmax


# Create a context_module
def context_module(input_image, filters, kernel_size=(3, 3), padding="same", strides=1):
    block = Conv2D(filters, kernel_size, strides, padding)(input_image)
    block = BatchNormalization()(block)
    block = Activation('relu')(block)
    block = Conv2D(filters, kernel_size, strides, padding)(block)
    block = BatchNormalization()(block)
    block = Activation('relu')(block)
    block = Dropout(rate=0.3)(block)
    return block


# Create a localization_module
def localization_module(input_image, filters):
    block = Conv2D(filters, kernel_size=(3, 3), padding="same", strides=1)(input_image)
    block = BatchNormalization()(block)
    block = Activation('relu')(block)
    block = Conv2D(filters, kernel_size=(1, 1), padding="same", strides=1)(block)
    block = BatchNormalization()(block)
    block = Activation('relu')(block)
    return block


# Create an upsampling_module
def upsampling_module(input_image, filters):
    block = UpSampling2D((2, 2))(input_image)
    block = Conv2D(filters, kernel_size=(3, 3), strides=1, padding="same")(block)
    block = BatchNormalization()(block)
    block = Activation('relu')(block)
    return block


# improved_Unet
def improved_Unet(input_image):
    # Contracting path
    enc1_1 = Conv2D(filters=16, kernel_size=(3, 3), padding="same", strides=1)(input_image)
    enc1_2 = context_module(enc1_1, filters=16)
    enc1 = add([enc1_1, enc1_2])

    enc2_1 = Conv2D(filters=32, kernel_size=(3, 3), padding="sane", strides=2)(enc1)
    enc2_2 = context_module(enc2_1, filters=32)
    enc2 = add([enc2_1, enc2_2])

    enc3_1 = Conv2D(filters=64, kernel_size=(3, 3), padding="sane", strides=2)(enc2)
    enc3_2 = context_module(enc3_1, filters=64)
    enc3 = add([enc3_1, enc3_2])

    enc4_1 = Conv2D(filters=128, kernel_size=(3, 3), padding="sane", strides=2)(enc3)
    enc4_2 = context_module(enc4_1, filters=128)
    enc4 = add([enc4_1, enc4_2])

    enc5_1 = Conv2D(filters=256, kernel_size=(3, 3), padding="sane", strides=2)(enc4)
    enc5_2 = context_module(enc5_1, filters=256)
    enc5_3 = add([enc5_1, enc5_2])
    enc5 = upsampling_module(enc5_3, filters=128)

    # Expansive path
    dec1_1 = Concatenate()([enc4, enc5])
    dec1_2 = localization_module(dec1_1, filters=128)
    dec1 = upsampling_module(dec1_2, filters=64)

    dec2_1 = Concatenate()([enc3, dec1])
    dec2_2 = localization_module(dec2_1, filters=64)
    dec2 = upsampling_module(dec2_2, filters=32)

    dec3_1 = Concatenate()([enc2_2, dec2])
    dec3_2 = localization_module(dec3_1, filters=32)
    dec3 = upsampling_module(dec3_2, filters=16)

    dec4_1 = Concatenate()([enc1_2, dec3])
    dec4_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same")(dec4_1)
    dec4 = Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding="same")(dec4_2)

    # Element-wise sum between segmentation layers
    seg1 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same")(dec2_2)
    seg1 = UpSampling2D((2, 2))(seg1)
    seg2 = Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding="same")(dec3_2)
    seg3 = add([seg1, seg2])
    seg3 = UpSampling2D((2, 2))(seg3)
    seg4 = Concatenate()([dec4, seg3])

    output = Activation('softmax')(seg4)
    model = Model(input_image, output)
    return model
