import tensorflow_addons 
import tensorflow as tf
from tensorflow.keras import models, layers, Model, Input
from tensorflow.keras.layers import Activation, Add, Conv2D 
from tensorflow.keras.layers import BatchNormalization, Dropout, LeakyReLU, UpSampling2D, MaxPooling2D, concatenate
tf.random.Generator = None
def upsample(layer_previous, size, activ_mode):   
    upsample_layer=UpSampling2D()(layer_previous)
    upsample_layer=Conv2D(size,(3,3),activation=activ_mode,padding="same")(upsample_layer)
    return upsample_layer

def context(layer_previous, size, activation=leakyReLu):     
    context=tensorflow_addons.layers.InstanceNormalization()(layer_previous)
    context=Activation(activation=activation)(context)
    context=Conv2D(size, (3, 3), padding="same")(context)
    context=Dropout(drop_prob)(context)
    context=tensorflow_addons.layers.InstanceNormalization()(context)
    context=Activation(activation=activation)(context)
    context=Conv2D(size, (3, 3), padding="same")(context)
    return context
    
def localization(layer_previous, size, activation=leakyReLu):         
    localization=Conv2D(size, (3, 3), activation = activation, padding="same")(layer_previous)        
    localization=Conv2D(size, (1, 1), activation = activation, padding="same")(localization)
    return localization
