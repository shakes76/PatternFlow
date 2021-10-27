##build all the models here##

#import stuff
#tensorflow stuff
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU, AveragePooling2D, Add, Input, UpSampling2D, Activation, Lambda, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import VarianceScaling
#other
from ModConv2D import ModConv2D
from functions import *

#style generator
def make_style_generator(latent_size):
    #standard deep NN
    model = tf.keras.Sequential()
    model.add(Dense(512, input_shape=[latent_size]))
    model.add(LeakyReLU(0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))

    return model

#synthesis network
def g_block(inputs, input_style, input_noise, im_size, filters, upsampling=True):
    
    #optional upsampling
    if upsampling:
        out = UpSampling2D(interpolation='bilinear')(inputs)
    else:
        out = Activation('linear')(inputs)

    #residual
    out_style = Dense(filters, kernel_initializer = VarianceScaling(200/out.shape[2]))(input_style)

    #main block
    #style stuff
    style = Dense(inputs.shape[-1], kernel_initializer = 'he_uniform')(input_style)
    delta = Lambda(crop_to_fit)([input_noise, out])
    d = Dense(filters, kernel_initializer='zeros')(delta)

    #ModConv2D block
    out = ModConv2D(filters, (3,3), strides=(1,1), padding="same", kernel_initializer = 'he_uniform')([out, style])
    out = Add()([out, d])
    out = LeakyReLU(0.2)(out)

    return out, to_output(out, out_style, im_size)


def make_synthesis_network (n_layers, im_size, batch_size, depth):
    #Inputs
    input_styles = []
    #input_noises = []
    for i in range(n_layers):
        input_styles.append(Input(shape=[512]))

    input_noise = Input(shape=[im_size, im_size, 1])
    outs = []


    x = tf.ones([batch_size, 1]) #c

    x = Dense(4*4*4*depth, activation='relu', kernel_initializer='random_normal')(x) #learned constant input vector
    x = Reshape([4, 4, 4*depth])(x) #a [4, 4, 4*depth] tensor --> to feed next layer of 4x4

    x, r = g_block(x, input_styles[1], input_noise, im_size, 64*depth, upsampling=False)    #4x4
    outs.append(r)
    x, r = g_block(x, input_styles[2], input_noise, im_size, 32*depth)                      #8x8
    outs.append(r)
    x, r = g_block(x, input_styles[3], input_noise, im_size, 16*depth)                      #16x16
    outs.append(r)
    x, r = g_block(x, input_styles[4], input_noise, im_size, 8*depth)                       #32x32
    outs.append(r)
    x, r = g_block(x, input_styles[5], input_noise, im_size, 4*depth)                       #64x64
    outs.append(r)
    x, r = g_block(x, input_styles[6], input_noise, im_size, 2*depth)                       #128x128
    outs.append(r)
    x, r = g_block(x, input_styles[7], input_noise, im_size, depth)                         #256x256
    outs.append(r)

    x = Add()(outs)

    #normalise
    x = Lambda(lambda y: y/2 + 0.5)(x)
    
    model = Model(inputs = [input_styles, input_noise], outputs = x)
    return model

#generator model
def make_generator_model(S, G, n_layers, latent_size, im_size):
    input_z = []
    W = []
    for i in range(n_layers):
        input_z.append(Input([latent_size]))
        W.append(S(input_z[-1]))

    input_noise = Input([im_size, im_size, 1])

    generated_image = G((W, input_noise))
    
    gen_model = Model(inputs = [input_z, input_noise], outputs = generated_image)

    return gen_model



#discriminator
def d_block(inputs, filters, pooling=True):
    residual = Conv2D(filters, 1)(inputs)

    out = Conv2D(filters, (3,3), padding='same')(inputs)
    out = LeakyReLU(0.2)(out)
    out = Conv2D(filters, (3,3), padding='same')(out)
    out = LeakyReLU(0.2)(out)
    
    out = Dropout(0.1)(out) #dropout helps with mode collapse

    out = Add()([residual, out])

    if pooling:
        out = AveragePooling2D()(out)

    return out

def make_discriminator_model(im_size, depth):
    inputs = Input(shape=(im_size, im_size, 1))

    x = d_block(inputs, depth)
    x = d_block(x, depth * 2)
    x = d_block(x, depth * 4)
    x = d_block(x, depth * 8)
    x = d_block(x, depth * 16, pooling=False)

    #classification stuff
    x = Flatten()(x)
    x = Dense(1)(x)

    model = Model(inputs= inputs, outputs = x)

    return model