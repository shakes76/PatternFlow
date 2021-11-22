"""
Architecture for Improved Unet

@author Lachlan Taylor
"""
from tensorflow import keras

"""
Layers for context module
"""
def context_module(last, dims):
    context = keras.layers.Conv2D(dims, kernel_size =3, padding = 'same')(last)
    context = keras.layers.BatchNormalization()(context)
    context = keras.layers.LeakyReLU(alpha=0.01)(context)
    context = keras.layers.Dropout(.3)(context)
    context = keras.layers.Conv2D(dims, kernel_size =3, padding = 'same')(context)
    context = keras.layers.BatchNormalization()(context)
    context = keras.layers.LeakyReLU(alpha=0.01)(context)

    return context

"""
Layers for segmentation
"""
def segmentation_block(x):
    seg = keras.layers.Conv2D(1, (1,1), activation = 'sigmoid')(x)
    return seg

"""
Describes a encoding block
"""
def encode_block(last, dims, activation_function, stride_num):
    conv = keras.layers.Conv2D(dims, (3,3), activation = activation_function, padding ='same', strides=stride_num)(last)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.LeakyReLU(alpha=0.01)(conv)
    context = context_module(conv, dims)
    sum = keras.layers.Add()([context, conv])
    return sum

"""
Layers for upsampling
"""
def upsample_block(last, dims, activation_function):
    up = keras.layers.UpSampling2D( size=(2, 2) )(last)
    conv = keras.layers.Conv2D(dims, (3,3), activation = activation_function, padding ='same')(up)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.LeakyReLU(alpha=0.01)(conv)
    return conv

"""
Layers for localisation
"""
def local_block(last, dims, activation_function):
    local = keras.layers.Conv2D(dims, (3,3), activation = activation_function, padding ='same')(last)
    local = keras.layers.BatchNormalization()(local)
    local = keras.layers.LeakyReLU(alpha=0.01)(local)
    local = keras.layers.Conv2D(dims, (1,1), activation = activation_function, padding ='same')(local)
    local = keras.layers.BatchNormalization()(local)
    local = keras.layers.LeakyReLU(alpha=0.01)(local)
    return local

"""
Initial layers for improved unet model
"""
def improved_unet(height, width):
    inputs = keras.layers.Input((height, width, 1))
    activation_function = keras.layers.LeakyReLU(alpha=0.01)

    # encoding
    sum1 = encode_block(inputs, 16, activation_function, 1)

    sum2 = encode_block(sum1, 32, activation_function, 2)

    sum3 = encode_block(sum2, 64, activation_function, 2)

    sum4 = encode_block(sum3, 128, activation_function, 2)

    sum5 = encode_block(sum4, 256, activation_function, 2)
    up1 = upsample_block(sum5, 128, activation_function)

    # decoding
    concat1 = keras.layers.concatenate([sum4,up1])
    local1 = local_block(concat1, 128, activation_function)
    up2 = upsample_block(local1, 64, activation_function)

    concat2 = keras.layers.concatenate([sum3,up2])
    local2 = local_block(concat2, 64, activation_function)
    up3 = upsample_block(local2, 32, activation_function)

    concat3 = keras.layers.concatenate([sum2,up3])
    local3 = local_block(concat3, 32, activation_function)
    up4 = upsample_block(local3, 16, activation_function)

    concat4 = keras.layers.concatenate([sum1,up4])
    conv_final = keras.layers.Conv2D(32, (3,3), activation = activation_function, padding ='same')(concat4)

    # segmenting
    seg1 = segmentation_block(local2)
    seg1 = keras.layers.UpSampling2D( size=(4, 4) )(seg1)
    seg2 = segmentation_block(local3)
    seg2 = keras.layers.UpSampling2D( size=(2, 2) )(seg2)
    seg3 = segmentation_block(conv_final)
    seg_sum = keras.layers.Add()([seg1, seg2, seg3])

    # result
    outputs = keras.layers.Conv2D(1, 1, activation = 'sigmoid')(seg_sum)
    model = keras.Model(inputs = inputs, outputs = outputs)
    return model