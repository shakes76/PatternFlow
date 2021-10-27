import tensorflow as tf
from tensorflow.keras import layers, models

def relu_conv_block(input_layer, conv_depth):
    conv = layers.Conv3D(conv_depth, (3, 3, 3), padding='same', activation ='relu') (input_layer)
    conv = layers.BatchNormalization()(conv)
    conv = layers.ReLU()(conv)

    return conv

def encoder_block(input_layer, conv_depth, drop_out=0.2):
    en = relu_conv_block(input_layer, conv_depth)
    en = layers.Dropout(drop_out)(en)
    en = relu_conv_block(en, conv_depth)
    return en

def decoder_block(input_layer, concat_layer, conv_depth, drop_out=0.2):
    de = layers.Conv3DTranspose(conv_depth, (2, 2, 2), strides=(2, 2, 2), padding='same')(input_layer)
    de = layers.concatenate([de, concat_layer])
    de = relu_conv_block(de, conv_depth)
    de = layers.Dropout(drop_out)(de)
    de = relu_conv_block(de, conv_depth)

    return de

def unet_model(row=256, col=256,height=128,segments = 6, channel = 3, conv_depth = 32):

    #input layer
    inputs = layers.Input(shape=(row, col, height, channel))
    #encoder path
    en1 = encoder_block(inputs,conv_depth)
    en2 = layers.MaxPooling3D(pool_size=(2,2,2))(en1)
    en2 = encoder_block(en2,conv_depth*2,0.3)
    en3 = layers.MaxPooling3D(pool_size=(2,2,2))(en2)
    en3 = encoder_block(en3,conv_depth*4,0.4)
    en4 = layers.MaxPooling3D(pool_size=(2,2,2))(en3)
    en4 = encoder_block(en4,conv_depth*8,0.4)
    en5 = layers.MaxPooling3D(pool_size=(2,2,2))(en4)
    en5 = encoder_block(en5,conv_depth*16,0.5)

    #decoder path
    de1 = decoder_block(en5,en4,conv_depth*8)
    de2 = decoder_block(de1, en3, conv_depth * 4, 0.3)
    de3 = decoder_block(de2,en2,conv_depth*2,0.4)
    de4 = decoder_block(de3,en1,conv_depth,0.5)

    #output layer
    outputs = layers.Conv3D(segments,(1,1,1),activation = 'sigmoid')(de4)
    model = models.Model(inputs,outputs)
    #compile model
    model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
    #print model summary
    model.summary()
    return model




