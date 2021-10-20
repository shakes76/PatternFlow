import tensorflow_addons 
import tensorflow as tf
from tensorflow.keras import models, layers, Model, Input
from tensorflow.keras.layers import Activation, Add, Conv2D 
from tensorflow.keras.layers import BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D, concatenate
tf.random.Generator = None
def upsample(layer_previous, size, activ_mode):
    '''
    Upsample the maps with lower resolution.
    The kernel size of the convolutional layer is 3*3.
    Return the upsampled layer.
    '''
    upsample_layer=UpSampling2D()(layer_previous)
    upsample_layer=Conv2D(size,(3,3),activation=activ_mode,padding="same")(upsample_layer)
    return upsample_layer

def improved_unet(n_classes, size=(256,256,3)):
    '''
    This is the main part of the improved Unet model.
    Return the improved Unet model.
    '''
    number_filter=16
    drop_prob=0.3
    leakyReLu=LeakyReLU(alpha=1e-2)
    
    def context(layer_previous, size, activation=leakyReLu):
        '''
        Context module's structure: a convolutional layer followed by a drop out layer and a convolutional layer.
        The kernel size of the convolutional layer is 3*3.
        The drop probability is 0.3 for drop out layers.
        Return the layer after entering the context module.
        '''
        context=tensorflow_addons.layers.InstanceNormalization()(layer_previous)
        context=Activation(activation=activation)(context)
        context=Conv2D(size, (3, 3), padding="same")(context)
        context=Dropout(drop_prob)(context)
        context=tensorflow_addons.layers.InstanceNormalization()(context)
        context=Activation(activation=activation)(context)
        context=Conv2D(size, (3, 3), padding="same")(context)
        return context
    
    def localization(layer_previous, size, activation=leakyReLu):
        '''
        localization module's structure: a convolutional layer (kernel size is 3*3) and another convolutional layer (kernel size is 1*1).
        Return the layer after entering the localization module.
        '''
        localization=Conv2D(size, (3, 3), activation = activation, padding="same")(layer_previous)        
        localization=Conv2D(size, (1, 1), activation = activation, padding="same")(localization)
        return localization

    init=Input(size)
    convo_layer1=Conv2D(number_filter*1, (3, 3), activation=leakyReLu, padding="same")(init)
    context1=context(convo_layer1, number_filter*1)
    sum1=Add()([convo_layer1, context1])
    convo_layer2=Conv2D(number_filter*2, (3, 3), strides=2, activation=leakyReLu, padding="same")(sum1)
    context2=context(convo_layer2, number_filter*2)
    sum2=Add()([convo_layer2, context2])
    convo_layer3=Conv2D(number_filter*4, (3, 3), strides=2, activation=leakyReLu, padding="same")(sum2)
    context3=context(convo_layer3, number_filter*4)
    sum3=Add()([convo_layer3, context3])
    convo_layer4=Conv2D(number_filter*8, (3, 3), strides=2, activation=leakyReLu, padding="same")(sum3)
    context4=context(convo_layer4, number_filter*8)
    sum4=Add()([convo_layer4, context4])
    convo_layer5=Conv2D(number_filter*16, (3, 3), strides=2, activation=leakyReLu, padding="same")(sum4)
    context5=context(convo_layer5, number_filter*16)
    sum5=Add()([convo_layer5, context5])

    upsample_1=upsample(sum5, number_filter*8, activ_mode = leakyReLu)
    concatenate_1=concatenate([sum4, upsample_1])
    localization_1=localization(concatenate_1, number_filter*8)
    upsample_2=upsample(localization_1, number_filter*4, activ_mode=leakyReLu)
    concatenate_2=concatenate([sum3, upsample_2])
    localization_2=localization(concatenate_2, number_filter*4)
    upsample_3=upsample(localization_2, number_filter*2, activ_mode=leakyReLu)
    concatenate_3=concatenate([sum2, upsample_3])

    #Segmentation layer 1
    seg1=Conv2D(1, (1, 1), activation=leakyReLu, padding="same")(localization_2)
    upper_level1=UpSampling2D(interpolation = "bilinear")(seg1)
    localization_3=localization(concatenate_3, number_filter*2)
    upsample_4=upsample(localization_3, number_filter*1, activ_mode=leakyReLu)
    concatenate_4=concatenate([sum1, upsample_4])
    #Segmentation layer 2
    seg2=Conv2D(1, (1, 1), activation=leakyReLu, padding="same")(localization_3)
    #Add segmentation layer 2 and upper level 1
    seg=Add()([seg2, upper_level1])
    #Upper
    upper_level2=UpSampling2D(interpolation="bilinear")(seg)
    #Last convolutional layer
    last_convo_layer=Conv2D(32, (3, 3), activation=leakyReLu, padding='same')(concatenate_4)
    #Last segmentation layer
    last_seg=Conv2D(n_classes, (1, 1), activation=leakyReLu, padding="same")(last_convo_layer) 
    final = Add()([last_seg, upper_level2])
    if n_classes>2:
        outputs=Activation('softmax')(final)
    else:
        outputs=Activation('sigmoid')(final)
    final_model=Model(inputs=init, outputs=outputs)
    return final_model
