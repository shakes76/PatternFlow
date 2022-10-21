from tensorflow.keras import models, layers, Model, Input
from tensorflow.keras.layers import Activation, Add, Conv2D, BatchNormalization, Dropout, LeakyReLU, UpSampling2D, MaxPooling2D, concatenate
import tensorflow_addons as tfa
import tensorflow as tf

tf.random.Generator = None

def upsampling_module(prev_layer, filter_size, activation):
    """ An upsampling module is upsampling the low resolution feature maps. It is done
        by means of a simple upscale that repeats the feature voxels twice in each spatial
        dimension, followed by a 3x3x3 convolution that halves the number of feature maps. 

    Args:
        prev_layer (tensorflow.keras.layers): previous layer (input layer to the upsampling module)
        filter_size (integer): size of the filter for convolutional layers
        activation (string/layers.Activation): activation function

    Returns:
        tensorflow.keras.layers: layer after getting through upsampling module
    """    
    
    upSamp = UpSampling2D()(prev_layer)
    upSamp = Conv2D(filter_size, (3,3), activation = activation, padding="same")(upSamp)

    return upSamp


def unet(num_output_classes, input_size=(256,256,3)):
    """ Model for the improved UNet from https://arxiv.org/abs/1505.04597

    Args:
        num_output_classes (integer): number of classes for the output
        input_size (tuple, optional): input image shape/size. Defaults to (256,256,3).

    Returns:
        tensorflow.keras.Model: a model of the improved UNet
    """  
    
    INIT_FILTER = 16
    hn = 'he_normal'
    dropout = 0.2

    inputs = Input(input_size)
    
    conv1 = Conv2D(INIT_FILTER * 1, (3,3), activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(INIT_FILTER * 1, (3,3), activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(INIT_FILTER * 2, (3,3), activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(INIT_FILTER * 2, (3,3), activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(INIT_FILTER * 4, (3,3), activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(INIT_FILTER * 4, (3,3), activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(INIT_FILTER * 8, (3,3), activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(INIT_FILTER * 8, (3,3), activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(INIT_FILTER * 16, (3,3), activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(INIT_FILTER * 16, (3,3), activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(dropout)(conv5)

    # up6 = (UpSampling2D(size = (2,2))(drop5))
    up6 = upsampling_module(drop5, INIT_FILTER * 8, 'relu')
    concat6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(INIT_FILTER * 8, (3,3), activation = 'relu', padding = 'same')(concat6)
    conv6 = Conv2D(INIT_FILTER * 8, (3,3), activation = 'relu', padding = 'same')(conv6)

    # up7 = (UpSampling2D(size = (2,2))(conv6))
    up7 = upsampling_module(conv6, INIT_FILTER * 4, 'relu')
    concat7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(INIT_FILTER * 4, (3,3), activation = 'relu', padding = 'same')(concat7)
    conv7 = Conv2D(INIT_FILTER * 4, (3,3), activation = 'relu', padding = 'same')(conv7)

    # up8 = (UpSampling2D(size = (2,2))(conv7))
    up8 = upsampling_module(conv7, INIT_FILTER * 2, 'relu')
    concat8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(INIT_FILTER * 2, (3,3), activation = 'relu', padding = 'same')(concat8)
    conv8 = Conv2D(INIT_FILTER * 2, (3,3), activation = 'relu', padding = 'same')(conv8)

    # up9 = (UpSampling2D(size = (2,2))(conv8))
    up9 = upsampling_module(conv8, INIT_FILTER * 1, 'relu')
    concat9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(INIT_FILTER * 1, (3,3), activation = 'relu', padding = 'same')(concat9)
    conv9 = Conv2D(INIT_FILTER * 1, (3,3), activation = 'relu', padding = 'same')(conv9)

    if num_output_classes > 2:
        print("SOFTMAX")
        outputs = Conv2D(num_output_classes, (1,1), activation = 'softmax', padding = 'same')(conv9)
    else:
        print("SIGMOID")
        outputs = Conv2D(num_output_classes, (1,1), activation = 'sigmoid', padding = 'same')(conv9) 
    model = Model(inputs = inputs, outputs = outputs)
    # model.summary()

    return model


def improved_unet(num_output_classes, input_size=(256,256,3)):
    """ Model for the improved UNet from https://arxiv.org/abs/1802.10508v1
        From the paper, they replaced the traditional batch with instance normalization 
        since they found that the stochasticity induced by their small batch sizes may 
        destabilize batch normalization.

    Args:
        num_output_classes (integer): number of classes for the output
        input_size (tuple, optional): input image shape/size. Defaults to (256,256,3).

    Returns:
        tensorflow.keras.Model: a model of the improved UNet
    """    

    INIT_FILTER = 16
    hn = 'he_normal'
    dropout = 0.3

    # Throughout the network we use leaky ReLU nonlinearities with a negative slope of 10^−2
    # for all feature map computing convolutions
    leakyReLu = LeakyReLU(alpha=1e-2)

    def context_module(prev_layer, filter_size, activation=leakyReLu):
        """ Each context module is in fact a pre-activation residual block with two
            3x3 convolutional layers and a dropout layer (pdrop = 0.3) in between. Context
            modules are connected by 3x3 convolutions with input stride 2 to reduce the
            resolution of the feature maps and allow for more features while descending down
            the aggregation pathway. 
            Ref: https://arxiv.org/abs/1603.05027

        Args:
            prev_layer (tensorflow.keras.layers): previous layer (input layer to the context module)
            filter_size (integer): size of the filter for convolutional layers
            activation (string/layers.Activation): activation function

        Returns:
            tensorflow.keras.layers: layer after getting through context module
        """        

        conMod = tfa.layers.InstanceNormalization()(prev_layer)
        conMod = Activation(activation=activation)(conMod)
        conMod = Conv2D(filter_size, (3, 3), padding="same")(conMod)
        conMod = Dropout(dropout)(conMod)

        conMod = tfa.layers.InstanceNormalization()(conMod)
        conMod = Activation(activation=activation)(conMod)
        conMod = Conv2D(filter_size, (3, 3), padding="same")(conMod)

        return conMod

    def localization_module(prev_layer, filter_size, activation=leakyReLu):
        """ A localization module consists of a 3x3x3 convolution followed by a 
            1x1x1 convolution that halves the number of feature maps.

        Args:
            prev_layer (tensorflow.keras.layers): previous layer (input layer to the localization module)
            filter_size (integer): size of the filter for convolutional layers
            activation (string/layers.Activation): activation function

        Returns:
            tensorflow.keras.layers: layer after getting through localization module
        """        
        # locMod = BatchNormalization()(prev_layer)
        locMod = Conv2D(filter_size, (3, 3), activation = activation, padding="same")(prev_layer)
        # locMod = BatchNormalization()(locMod)
        locMod = Conv2D(filter_size, (1, 1), activation = activation, padding="same")(locMod)
        
        return locMod

    ################################
    ###### Building the model ######
    ################################

    inputs = Input(input_size)

    # Encoder path, [16,32,64,128,256]
    conv1 = Conv2D(INIT_FILTER * 1, (3, 3), activation = leakyReLu, padding="same")(inputs)
    conMod1 = context_module(conv1, INIT_FILTER * 1)
    add1 = Add()([conv1, conMod1])

    conv2 = Conv2D(INIT_FILTER * 2, (3, 3), strides = 2, activation = leakyReLu, padding="same")(add1)
    conMod2 = context_module(conv2, INIT_FILTER * 2)
    add2 = Add()([conv2, conMod2])

    conv3 = Conv2D(INIT_FILTER * 4, (3, 3), strides = 2, activation = leakyReLu, padding="same")(add2)
    conMod3 = context_module(conv3, INIT_FILTER * 4)
    add3 = Add()([conv3, conMod3])

    conv4 = Conv2D(INIT_FILTER * 8, (3, 3), strides = 2, activation = leakyReLu, padding="same")(add3)
    conMod4 = context_module(conv4, INIT_FILTER * 8)
    add4 = Add()([conv4, conMod4])

    conv5 = Conv2D(INIT_FILTER * 16, (3, 3), strides = 2, activation = leakyReLu, padding="same")(add4)
    conMod5 = context_module(conv5, INIT_FILTER * 16)
    add5 = Add()([conv5, conMod5])

    # Decoder path. [128, 64, 32, 16]
    up1 = upsampling_module(add5, INIT_FILTER * 8, activation = leakyReLu)
    concat1 = concatenate([add4, up1])

    locMod1 = localization_module(concat1, INIT_FILTER * 8)
    up2 = upsampling_module(locMod1, INIT_FILTER * 4, activation = leakyReLu)
    concat2 = concatenate([add3, up2])

    locMod2 = localization_module(concat2, INIT_FILTER * 4)
    up3 = upsampling_module(locMod2, INIT_FILTER * 2, activation = leakyReLu)
    concat3 = concatenate([add2, up3])
    # segmentation layer 1
    segment1 = Conv2D(1, (1, 1), activation = leakyReLu, padding="same")(locMod2)
    upscale1 = UpSampling2D(interpolation = "bilinear")(segment1)

    locMod3 = localization_module(concat3, INIT_FILTER * 2)
    up4 = upsampling_module(locMod3, INIT_FILTER * 1, activation = leakyReLu)
    concat4 = concatenate([add1, up4])
    # segmentation layer 2
    segment2 = Conv2D(1, (1, 1), activation = leakyReLu, padding="same")(locMod3)
    # add segmentation layer 2 and upscale 1
    segment1_2 = Add()([segment2, upscale1])
    # upscale
    upscale2 = UpSampling2D(interpolation = "bilinear")(segment1_2)

    # final conv layer
    conv_final = Conv2D(32, (3, 3), activation = leakyReLu, padding='same')(concat4)
    # final segmentation layer
    segment_final = Conv2D(num_output_classes, (1, 1), activation = leakyReLu, padding="same")(conv_final) 
    add_final = Add()([segment_final, upscale2])

    if num_output_classes > 2:
        print("Activation function: softmax")
        outputs = Activation('softmax')(add_final)
    else:
        print("Activation function: sigmoid")
        outputs = Activation('sigmoid')(add_final)

    model = Model(inputs = inputs, outputs = outputs)

    return model
    