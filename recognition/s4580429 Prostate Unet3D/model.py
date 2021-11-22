"""
Functions for building and returning a 3D UNet model

@author Cody Baldry
@student_number 45804290
@date 1 November 2021
"""
from tensorflow import keras as K

# convolution block described in unet papers
def convBlock(model, name, filters, params):
    model = K.layers.Conv3D(filters=filters, **params, name=name+"_conv0")(model)
    model = K.layers.BatchNormalization(name=name+"_bn0")(model)
    model = K.layers.Activation("relu", name=name+"_relu0")(model)

    model = K.layers.Conv3D(filters=filters, **params, name=name+"_conv1")(model)
    model = K.layers.BatchNormalization(name=name+"_bn1")(model)
    model = K.layers.Activation("relu", name=name)(model)

    return model

# 3d unet model
def unet_3d(input_shape, n_classes, activation='softmax'):
    starting_filters = 32

    params = dict(kernel_size=(3,3,3), padding="same",
            kernel_initializer="he_uniform")
    params_trans = dict(kernel_size=(2,2,2), strides=(2,2,2),
            padding="same")
    p_size = (2,2,2)

    inputs = K.layers.Input(shape=input_shape, name="input")

    # encoding
    conv1 = convBlock(inputs, "down1", starting_filters, params)
    pool = K.layers.MaxPooling3D(name="pool1", pool_size=p_size)(conv1)

    conv2 = convBlock(pool, "down2", starting_filters*2, params)
    pool = K.layers.MaxPooling3D(name="pool2", pool_size=p_size)(conv2)

    conv3 = convBlock(pool, "down3", starting_filters*4, params)
    pool = K.layers.MaxPooling3D(name="pool3", pool_size=p_size)(conv3)

    conv4 = convBlock(pool, "down4", starting_filters*8, params)
    pool = K.layers.MaxPooling3D(name="pool4", pool_size=p_size)(conv4)

    encodeE = convBlock(pool, "down5", starting_filters*16, params)

    # decoding
    up = K.layers.Conv3DTranspose(name="tconv1", filters=starting_filters*8,
            **params_trans)(encodeE)
    concat = K.layers.concatenate([up, conv4], name="concat1")

    decodeC = convBlock(concat, "up1", starting_filters*8, params)

    up = K.layers.Conv3DTranspose(name="tconv2", filters=starting_filters*4,
            **params_trans)(decodeC)
    concat = K.layers.concatenate([up, conv3], name="concat2")

    decodeB = convBlock(concat, "up2", starting_filters*4, params)

    up = K.layers.Conv3DTranspose(name="tconv3", filters=starting_filters*2,
            **params_trans)(decodeB)
    concat = K.layers.concatenate([up, conv2], name="concat3")

    decodeA = convBlock(concat, "up3", starting_filters*2, params)

    up = K.layers.Conv3DTranspose(name="tconv4", filters=starting_filters,
            **params_trans)(decodeA)
    concat = K.layers.concatenate([up, conv1], name="concat4")

    
    convOut = convBlock(concat, "convOut", starting_filters, params)

    predict = K.layers.Conv3D(name="predict", filters=n_classes,
            kernel_size=(1,1,1), activation=activation)(convOut)

    model = K.models.Model(inputs=[inputs], outputs=[predict], name="unet3D")

    return model