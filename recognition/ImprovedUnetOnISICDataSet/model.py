import tensorflow as tf
import numpy as np

"""
Conducts a context module for the Imrpoved UNET. See Figure 1 in README.md
for it's application within the model.
Parameters: 
    input: input tensor that goes into the context module
    channels: Number of filters to perform the context module at
Returns:
    The resulting tensor of a context module to continue the overall model.
"""
def context_layer(input, channels):
    context_1 = tf.keras.layers.Conv2D(channels, (3, 3), padding="same")(input)
    activation1 = tf.keras.layers.LeakyReLU(alpha=0.3)(context_1)
    context_dropout = tf.keras.layers.Dropout(0.3)(activation1)
    context_2 = tf.keras.layers.Conv2D(channels, (3, 3), padding="same")(context_dropout)
    activation2 = tf.keras.layers.LeakyReLU(alpha=0.3)(context_2)
    return activation2

"""
Conducts a decoding step for the model, this is the 3x3 convilution, followed by
a context module, followed by an element-wise summation. See the original paper for
the Improved UNET model.
Parameters: 
    input: input tensor that goes into the context module
    channels: Number of filters to perform the context module at
Returns:
    The resulting tensor of a decode module to continue the overall model.
"""
def decode_layer(input, channels):
    decode_1 = tf.keras.layers.Conv2D(channels, (3, 3), strides=(2,2), padding="same")(input)
    activation1 = tf.keras.layers.LeakyReLU(alpha=0.3)(decode_1)
    context = context_layer(activation1, channels)
    elem_wise_sum = decode_1 + context
    return elem_wise_sum

"""
Conducts an upsampling module for the Improved UNET model. See Figure 1 in README.md
for it's application within the model.
Parameters: 
    input: input tensor that goes into the context module
    channels: Number of filters to perform the context module at
Returns:
    The resulting tensor of a upsampling module to continue the overall model.
"""
def upsampling_module(input, channels):
    up = tf.keras.layers.UpSampling2D()(input)
    conv2d = tf.keras.layers.Conv2D(channels, (3, 3), padding="same")(up)
    activation = tf.keras.layers.LeakyReLU(alpha=0.3)(conv2d)
    return activation

"""
Conducts a localisation module within the Improved UNET model. See Figure 1 in README.md
for it's application within the model.
Parameters: 
    input: input tensor that goes into the context module
    channels: Number of filters to perform the context module at
Returns:
    The resulting tensor of a localisation module to continue the overall model.
"""
def localization_layer(input, channels):
    conv2D_3 = tf.keras.layers.Conv2D(channels, (3, 3), padding="same")(input)
    activation1 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv2D_3)
    conv2D_1 = tf.keras.layers.Conv2D(channels, (1, 1), padding="same")(conv2D_3)
    activation2 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv2D_1)
    return activation2

"""
Conducts a segmentation layer for the model. See Figure 1 in README.md
for it's application within the model.
Parameters: 
    input: input tensor that goes into the context module
    channels: Number of filters to perform the context module at
Returns:
    The resulting tensor of a segmentation module to continue the overall model.
"""
def segmentation_layer(input, channels):
    conv2d = tf.keras.layers.Conv2D(channels, (3, 3), padding="same")(input)
    return tf.keras.layers.LeakyReLU(alpha=0.3)(conv2d)

"""
Constructs an Improved UNET model based on the number of output channels
for segmenation, the number of initial filters to conduct convolutions with
and the shape of the image. Note that the Green comments correspond with the
filter size of the original Improved UNET structure.
Parameters: 
    output_channels: number of segmentation layers
    f: number of filters to intitially start the model with. Initialised to
       16 as per Figure 1 in README.md
    input_shape: shape of the image recieved by the model.
Returns:
    An Improved UNET model that can be compiled.
"""
def improved_unet(output_channels, f=16, input_shape=(256, 256, 1)):
    modelInput = tf.keras.layers.Input(shape=(256, 256, 1))

    conv2D16_1 = tf.keras.layers.Conv2D(f, (3, 3), padding="same")(modelInput)
    activation1 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv2D16_1)

    #1st Context Layer
    context_16 = context_layer(activation1, f)

    elem_wise_sum_16 = activation1 + context_16

    #Complete the Decoding phase of the UNET
    d1 = decode_layer(elem_wise_sum_16, 2*f)

    d2 = decode_layer(d1, 4*f)

    d3 = decode_layer(d2, 8*f)

    d4 = decode_layer(d3, 16*f)

    #Start of the Encoding phase of the UNET
    u128 = upsampling_module(d4, 8*f)

    #Concatenation with 128 filters
    concat1 = tf.keras.layers.Concatenate()([u128, d3])

    local_layer_128 = localization_layer(concat1, 8*f)

    u64 = upsampling_module(local_layer_128, 4*f)

    #Concatenation with 64 filters
    concat2 = tf.keras.layers.Concatenate()([u64, d2])

    local_layer_64 = localization_layer(concat2, 4*f)

    #Segmentation with 64 filters
    seg64 = segmentation_layer(local_layer_64, output_channels)

    u32 = upsampling_module(local_layer_64, 2*f)

    #Concatenation with 32 filters
    concat3 = tf.keras.layers.Concatenate()([u32, d1])

    local_layer_32 = localization_layer(concat3, 2*f)

    #Segmenation with 32 filters
    seg32 = segmentation_layer(local_layer_32, output_channels)

    u16 = upsampling_module(local_layer_32, f)

    #Concatenation with 16 filters
    concat4 = tf.keras.layers.Concatenate()([u16, elem_wise_sum_16])

    #Final 32 filters convolution
    conv2d_32 = tf.keras.layers.Conv2D(2*f, (3, 3), padding="same")(concat4)

    activation32 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv2d_32)

    #Segmentation with 32 filters
    seg32_1 = segmentation_layer(activation32, output_channels)

    seg64_upscaled = tf.keras.layers.UpSampling2D()(seg64)

    elemwise_sum_seg1 = seg64_upscaled + seg32

    seg32_upscaled = tf.keras.layers.UpSampling2D()(elemwise_sum_seg1)

    #Pre segment layers from 64 filters upscaled to then 32 filters
    segment = seg32_1 + seg32_upscaled

    #Final segment layer.
    output = tf.keras.layers.Conv2D(output_channels, (3, 3), activation="softmax", padding="same")(segment)

    model = tf.keras.Model(inputs=modelInput, outputs=output)
    return model