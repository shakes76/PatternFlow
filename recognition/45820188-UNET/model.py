from tensorflow.keras.layers import Input, Conv2D, Dropout, LeakyReLU, UpSampling2D
from tensorflow.keras.layers import Concatenate, Add, Dense
from tensorflow.keras.models import Model

# Building the Model
def conv3x3(input_layer, filters, kernel_size=(3,3), strides=(1,1)):
    """
    Base 3x3x3 convolution with default kernal 3 and stride 1

    Params:
        input_layer: The starting layer given to the convolution 
        filters: Size of the filters
        kernel_size: Size of the kernel given
        strides: Stride size

    Return:
        Returns a layer of convolution into a Leaky ReLU activation
    """
    conv1 = Conv2D(filters, kernel_size, strides, padding="same", kernel_initializer="he_normal")(input_layer) 
    # maybe do a batch norm here too?
    # https://edstem.org/au/courses/6584/discussion/652021
    return LeakyReLU(0.01)(conv1)

def context_module(input_layer, filters):
    """
    Context module, does two convolutions, with a dropout between

    Params:
        input_layer: The starting layer given to the convolution
        filter: Size of the filters
    
    Return: 
        Returns a context module, which is convolution into dropout
        then another convolution
    """
    conv1 = conv3x3(input_layer, filters)
    drop1 = Dropout(0.3)(conv1)
    return conv3x3(drop1, filters)

def upsampling_module(input_layer, filters):
    """
    Upsampling Module which includes upsampling then convolution

    Params:
        input_layer: The starting layer given to the convolution
        filter: Size of the filters

    Return:
        Returns an upsampling module through an upsampling layer
        then a convolution layer
    """
    upsample1 = UpSampling2D(size=(2,2))(input_layer)
    return conv3x3(upsample1, filters)

def localisation_module(input_layer, filters):
    """
    Localisation Module 

    Params:
        input_layer: The starting layer given to the convolution
        filter: Size of the filters

    Return: 
        Returns localisation module, which is two convolutions
        with a second kernel size of 1
    """
    conv1 = conv3x3(input_layer, filters)
    return conv3x3(conv1, filters, kernel_size=(1,1))

def build_model(input_shape, depth):
    """
    Source: https://arxiv.org/pdf/1802.10508v1.pdf

    Params: 
        input_shape: The shape of the input images given 
        depth: Size of starting convolution layer
    
    Return:
        Returns an Improved UNET Model
    """
    inputs = Input(input_shape)

    # Each of the layers for the first half of this UNET model
    # require a 3x3x3 convolution into a context module, with
    # an Add of both these together. The first convolution has
    # stride 1, whereas the following layers have stride 2
    
    # First Context Level
    conv1 = conv3x3(inputs, depth)
    context1 = context_module(conv1, depth)
    concat1 = Add()([conv1, context1])
    
    # Second Context Level
    conv2 = conv3x3(concat1, depth*2, strides=(2,2))
    context2 = context_module(conv2, depth*2)
    concat2 = Add()([conv2, context2])
    
    # Third Context Level
    conv3 = conv3x3(concat2, depth*4, strides=(2,2))
    context3 = context_module(conv3, depth*4)
    concat3 = Add()([conv3, context3])
    
    # Fourth Context Level
    conv4 = conv3x3(concat3, depth*8, strides=(2,2))
    context4 = context_module(conv4, depth*8)
    concat4 = Add()([conv4, context4])
    
    # Fifth (Last) Context Level
    conv5 = conv3x3(concat4, depth*16, strides=(2,2))
    context5 = context_module(conv5, depth*16)
    concat5 = Add()([conv5, context5])

    # This is where upsampling begins
    # Upsample the layer and the concatenate it with the 
    # concatenate of the first half of the UNET, named
    # concatN, where N is the layer number

    upsample1 = upsampling_module(concat5, depth*8)
    save1 = Concatenate()([upsample1, concat4])    
    localise1 = localisation_module(save1, depth*8)

    upsample2 = upsampling_module(localise1, depth*4)
    save2 = Concatenate()([upsample2, concat3])
    localise2 = localisation_module(save2, depth*4)
    
    seg1 = conv3x3(localise2, 16, kernel_size=(1,1))
    seg1 = UpSampling2D(size=(2,2))(seg1)
    
    upsample3 = upsampling_module(localise2, depth*2)
    save3 = Concatenate()([upsample3, concat2])
    localise3 = localisation_module(save3, depth*2)

    # For the last few layers, a segmentation layer is taken
    # then a final 3x3x3 convolution with stride 1
    
    seg2 = conv3x3(localise3, 16)
    seg2 = Add()([seg2, seg1])
    seg2 = UpSampling2D(size=(2,2))(seg2)
    
    upsample4 = upsampling_module(localise3, depth)
    save4 = Concatenate()([upsample4, concat1])
    
    conv_last = conv3x3(save4, depth*2)
    seg3 = conv3x3(conv_last, depth)
    seg3 = Add()([seg3, seg2])
    
    # Final Softmax as given by the paper
    softmax = Conv2D(1, kernel_size=(1,1), padding="same", activation="softmax")(seg3)

    # Sigmoid to ensure the output is between 0 and 1
    sigmoid = Dense(1, activation="sigmoid")(softmax)
    
    outputs = sigmoid
    model = Model(inputs, outputs)
    model.summary()
    
    return model 