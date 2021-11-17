import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow_addons.layers import InstanceNormalization

def convolution(inputs, filters,stride):
    # function to implement convolution layer
    conv = tf.keras.layers.Conv2D(filters, (3, 3), strides = stride, padding='same', activation='relu')(inputs)
    return conv

def context_module(inputs,filters):
    # the context module in UNet module.
    inputs_norm = InstanceNormalization()(inputs)
    output = tf.keras.layers.Conv2D(filters,(3, 3), activation= LeakyReLU(alpha=0.01), padding='same')(inputs_norm)
    output = Dropout(0.3)(output)
    output = InstanceNormalization()(output)
    output = tf.keras.layers.Conv2D(filters, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(output)

    final_output = inputs + output

    return final_output


def upsample(inputs,filter):
    #  function to implement unsample layer.
    conv = tf.keras.layers.UpSampling2D(size=(2,2))(inputs)
    conv = tf.keras.layers.Conv2D(filter, (3,3), strides= 1, activation = LeakyReLU(alpha=0.01),  padding="same")(conv)
    return conv

def localization_module(inputs, filters):
    # the localization module in UNet module.
    conv1 = tf.keras.layers.Conv2D(filters,(3, 3), activation= LeakyReLU(alpha=0.01), padding='same')(inputs)
    norm1 = InstanceNormalization()(conv1)

    conv2 = tf.keras.layers.Conv2D(filters,(3, 3), activation= LeakyReLU(alpha=0.01), padding='same')(norm1)
    norm2 = InstanceNormalization()(conv2)

    return norm1, norm2


def Unet():
    # build the Unet model.

    inputs = tf.keras.layers.Input(shape=(256, 256, 3))


    # context section

    conv_1 = convolution(inputs, 4, 1)
    con1_context = context_module(conv_1,4)

    conv_2 = convolution(con1_context, 8, 2)
    con2_context = context_module(conv_2,8)

    conv_3 = convolution(con2_context, 16, 2)
    con3_context = context_module(conv_3,16)

    conv_4 = convolution(con3_context, 32,2)
    con4_context = context_module(conv_4, 32)

    conv_5 = convolution(con4_context, 64, 2)
    con5_context = context_module(conv_5, 64)


    # localization

    local_1_conv = upsample(con5_context,32)

    local_2_concat = concatenate([local_1_conv, con4_context])

    norm, local_2_localization = localization_module(local_2_concat,32)

    local_2_conv = upsample(local_2_localization,16)



    local_3_concat = concatenate([local_2_conv, con3_context])

    norm_1,local_3_localization = localization_module(local_3_concat, 16)

    local_3_conv = upsample(local_3_localization, 8)


    local_4_concat = concatenate([local_3_conv, con2_context])

    norm_2,local_4_localization = localization_module(local_4_concat, 8)

    local_4_conv = upsample(local_4_localization, 4)


    final_concat = concatenate([local_4_conv,con1_context])
    final_conv = tf.keras.layers.Conv2D(8,(3, 3), activation= LeakyReLU(alpha=0.01), padding='same')(final_concat)

    # segmentation section

    seg_1 = Conv2D(2,(1,1))(norm_1)
    seg_1_up = UpSampling2D(interpolation='bilinear')(seg_1)

    seg_2 = Conv2D(2, (1, 1))(norm_2)
    seg_2_up = UpSampling2D(interpolation='bilinear')(seg_2+seg_1_up)

    seg_3 = Conv2D(2, (1, 1))(final_conv)
    final_conv = seg_2_up + seg_3

    output = Activation('sigmoid')(final_conv)

    return Model(inputs=inputs,outputs=output)
