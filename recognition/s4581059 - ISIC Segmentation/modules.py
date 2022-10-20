#Contains the source code of the components of the model.

import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, UpSampling2D, Concatenate, Add
from keras.models import Model

#"Throughout the network we use leaky ReLU nonlinearities with a negative slopes of 10^-2" - [1]
relu = tf.keras.layers.LeakyReLU(alpha=1e-2)
kernel_size = (3,3)

def conv_block(input, num_filters):
    """
    Creates a convolutional block - i.e. Two convolutional layers with a dropout layer in between
    Noted: Dropout layer rate is set to 0.3
    Param: input - the layer in which the convolutional block will input
    Param: num_filters - the number of filters required for the convolutional block
    Returns: The outputted convolutional layer
    """
    conv_1 = Conv2D(num_filters, kernel_size, padding="same", activation=relu)(input)
    drop = Dropout(0.3)(conv_1)
    conv_2 = Conv2D(num_filters, kernel_size, padding="same", activation=relu)(drop)
    
    return conv_2

def localisation_block(input, num_filters):
    """
    Creates a localisation block - i.e. Two convolutional layers which decrease the kernel size
    Param: input - the layer in which the convolutional block will input
    Param: num_filters - the number of filters required for the convolutional block
    Returns: The outputted convolutional layer
    """
    conv_1 = Conv2D(num_filters, kernel_size, padding="same", activation=relu)(input)
    conv_2 = Conv2D(num_filters, (1, 1), padding="same", activation=relu)(conv_1)
    return conv_2

def model():
    """
    Model made up of an encoder, decoder and the concatenation of the 2
    """
    
    
    #Starting number of filters
    filters = 16

    ##Encoder
    #Encoder consists of 2 3x3x3 convolutional layers with a dropout layer between them 
    #16 filters
    input = Input(shape=(128, 128, 1))
    layer_1 = Conv2D(filters, kernel_size, padding="same", activation=relu)(input)
    block_1 = conv_block(layer_1, filters)
    add_block_1 = Add()([layer_1, block_1])
    
    #Context modules are connected by 3x3x3 convolutions with input stride 2
    #32 filters
    layer_2 = Conv2D(filters * 2, kernel_size, strides=(2, 2), padding="same", activation=relu)(add_block_1)
    block_2 = conv_block(layer_2, filters * 2)
    add_block_2 = Add()([layer_2, block_2])

    #64 filters
    layer_3 = Conv2D(filters * 4, kernel_size, strides=(2, 2), padding="same", activation=relu)(add_block_2)
    block_3 = conv_block(layer_3, filters * 4)
    add_block_3 = Add()([layer_3, block_3])

    #128 filters
    layer_4 = Conv2D(filters * 8, kernel_size, strides=(2, 2), padding="same", activation=relu)(add_block_3)
    block_4 = conv_block(layer_4, filters * 8)
    add_block_4 = Add()([layer_4, block_4])

    #256 filters
    layer_5 = Conv2D(filters * 16, kernel_size, strides=(2, 2), padding="same", activation=relu)(add_block_4)
    block_5 = conv_block(layer_5, filters * 16)
    add_block_5 = Add()([layer_5, block_5])

    ##Decoder
    #Base level
    dec_1_upsample = UpSampling2D()(add_block_5)
    dec_1_conv = Conv2D(filters * 8, kernel_size, padding="same", activation=relu)(dec_1_upsample)
    dec_1_concat = Concatenate()([add_block_4, dec_1_conv])

    #128 filters Local
    dec_2_conv_1 = localisation_block(dec_1_concat, filters * 8)
    dec_2_upsample = UpSampling2D()(dec_2_conv_1)
    dec_2_conv = Conv2D(filters * 4, kernel_size, padding="same", activation=relu)(dec_2_upsample)
    dec_2_concat = Concatenate()([add_block_3, dec_2_conv])

    #64 filters Local
    dec_3_conv_1 = localisation_block(dec_2_concat, filters * 4)
    dec_3_upsample = UpSampling2D()(dec_3_conv_1)
    dec_3_conv = Conv2D(filters * 2, kernel_size, padding="same", activation=relu)(dec_3_upsample)
    dec_3_concat = Concatenate()([add_block_2, dec_3_conv])

    #32 filters Local
    dec_4_conv_1 = localisation_block(dec_3_concat, filters * 2)
    dec_4_upsample = UpSampling2D()(dec_4_conv_1)
    dec_4_conv = Conv2D(filters, kernel_size, padding="same", activation=relu)(dec_4_upsample)
    dec_4_concat = Concatenate()([add_block_1, dec_4_conv])

    #3x3 Convolution
    dec_5_conv_1 = Conv2D(filters * 2, kernel_size, padding="same", activation=relu)(dec_4_concat)

    ##Segmentation Layers
    #General idea 
    # 1. Create segmentation layer - at appropriate height
    # 2. Add to lower segmentation layer where appropriate
    # 3. Upscale

    seg_1 = Conv2D(filters, (1, 1), padding="same", activation=relu)(dec_3_conv_1)
    #Bottom layer therefore skip adding
    seg_1 = UpSampling2D()(seg_1)

    seg_2 = Conv2D(filters, (1, 1), padding="same", activation=relu)(dec_4_conv_1)
    seg_2 = Add()([seg_2, seg_1])
    seg_2 = UpSampling2D()(seg_2)

    seg_3 = Conv2D(filters, (1, 1), padding="same", activation=relu)(dec_5_conv_1)
    seg_3 = Add()([seg_3, seg_2])
    #Last segmentation layer therefore no need to upscale

    ##Softmax
    output = Conv2D(2, kernel_size, padding= "same", activation="softmax")(seg_3)
    
    model = Model(input, output)
    return model   