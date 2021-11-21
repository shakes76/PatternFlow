import tensorflow_addons 
import tensorflow as tf
from tensorflow.keras import models, layers, Model, Input
from tensorflow.keras.layers import Activation, Add, Conv2D 
from tensorflow.keras.layers import BatchNormalization, Dropout, LeakyReLU, UpSampling2D, MaxPooling2D, concatenate

def upsample(layer_previous, size, activation):   
    upsample_layer=UpSampling2D()(layer_previous)
    upsample_layer=Conv2D(size,(3,3),activation=activ_mode,padding="same")(upsample_layer)
    return upsample_layer

def basic_unet(n_classes,size=(256,256,3)):
    number_filter=16
    drop_prob = 0.2
    init = Input(size)
    convo_layer1=Conv2D(number_filter*1, (3,3), activation='relu', padding='same')(init)
    convo_layer1=Conv2D(number_filter*1, (3,3), activation='relu', padding='same')(convo_layer1)
    max_pool1=MaxPooling2D(pool_size=(2, 2))(convo_layer1)
    convo_layer2=Conv2D(number_filter*2, (3,3), activation='relu', padding='same')(max_pool1)
    convo_layer2=Conv2D(number_filter*2, (3,3), activation='relu', padding='same')(convo_layer2)
    max_pool2=MaxPooling2D(pool_size=(2, 2))(convo_layer2)
    convo_layer3=Conv2D(number_filter*4, (3,3), activation='relu', padding='same')(max_pool2)
    convo_layer3=Conv2D(number_filter*4, (3,3), activation='relu', padding='same')(convo_layer3)
    max_pool3=MaxPooling2D(pool_size=(2, 2))(convo_layer3)
    convo_layer4=Conv2D(number_filter*8, (3,3), activation='relu', padding='same')(max_pool3)
    convo_layer4=Conv2D(number_filter*8, (3,3), activation='relu', padding='same')(convo_layer4)
    drop_layer4=Dropout(drop_prob)(convo_layer4)
    max_pool4=MaxPooling2D(pool_size=(2, 2))(drop_layer4)
    convo_layer5=Conv2D(number_filter*16, (3,3), activation='relu', padding='same')(max_pool4)
    convo_layer5=Conv2D(number_filter*16, (3,3), activation='relu', padding='same')(convo_layer5)
    drop_layer5=Dropout(drop_prob)(convo_layer5)
    upsample_6=upsample(drop_layer5, number_filter*8, 'relu')
    concatenate_6=concatenate([drop_layer4,upsample_6], axis=3)
    convo_layer6=Conv2D(number_filter*8, (3,3), activation='relu', padding='same')(concatenate_6)
    convo_layer6=Conv2D(number_filter*8, (3,3), activation='relu', padding='same')(convo_layer6)
    upsample_7=upsample(convo_layer6, number_filter*4, 'relu')
    concatenate_7=concatenate([convo_layer3,upsample_7], axis=3)
    convo_layer7=Conv2D(number_filter*4, (3,3), activation='relu', padding='same')(concatenate_7)
    convo_layer7=Conv2D(number_filter*4, (3,3), activation='relu', padding='same')(convo_layer7)
    upsample_8=upsample(convo_layer7, number_filter*2, 'relu')
    concatenate_8=concatenate([conv2,upsample_8], axis=3)
    convo_layer8=Conv2D(number_filter*2, (3,3), activation='relu', padding='same')(concatenate_8)
    convo_layer8=Conv2D(number_filter*2, (3,3), activation='relu', padding='same')(convo_layer8)
    upsample_9=upsample(convo_layer8, number_filter*1, 'relu')
    concatenate_9=concatenate([convo_layer1,upsample_9], axis = 3)
    convo_layer9=Conv2D(number_filter*1, (3,3), activation='relu', padding='same')(concatenate_9)
    convo_layer9=Conv2D(number_filter*1, (3,3), activation='relu', padding='same')(convo_layer9)
    if n_classes>2:
        outputs=Conv2D(n_classes,(1,1),activation='softmax', padding='same')(convo_layer9)
    else:
        outputs=Conv2D(n_classes,(1,1), activation='sigmoid', padding='same')(convo_layer9) 
    final_model = Model(inputs=init, outputs=outputs)
    return final_model
