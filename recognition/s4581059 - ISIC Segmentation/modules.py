#Contains the source code of the components of the model.

import tensorflow as tf
from keras.layers import Input, Conv3D, MaxPooling3D, Dropout, Dense, UpSampling3D, Concatenate, Add
from keras.models import Model


def model():
    """
    Model made up of an encoder, decoder and the concatenation of the 2
    """
    #"Throughout the network we use leaky ReLU nonlinearities with a negative slopes of 10^-2" - [1]
    relu = tf.keras.layers.LeakyReLU(alpha=1e-2)
    kernel_size = (3,3,3)
    filters = 16

    #Encoder consists of 2 3x3x3 convolutional layers with a dropout layer between them 
    #16 filters
    input = Input(shape=(64, 64, 16, 1))
    block_1_conv_1 = Conv3D(filters, kernel_size, padding= "same", activation=relu)(input)
    block_1_dropout = Dropout(0.3)(block_1_conv_1)
    block_1_conv_2 = Conv3D(filters, kernel_size, padding= "same", activation=relu)(block_1_dropout)
    add_block_1 = Add()([block_1_conv_1, block_1_conv_2])
    
    #Context modules are connected by 3x3x3 convolutions with input stride 2
    #32 filters
    block_2_conv_1 = Conv3D(filters * 2, kernel_size, strides=(2, 2, 2), padding= "same", activation=relu)(add_block_1)
    block_2_dropout = Dropout(0.3)(block_2_conv_1)
    block_2_conv_2 = Conv3D(filters * 2, kernel_size, padding= "same", activation=relu)(block_2_dropout)
    add_block_2 = Add()([block_2_conv_1, block_2_conv_2])

    #64 filters
    block_3_conv_1 = Conv3D(filters * 4, kernel_size, strides=(2, 2, 2), padding= "same", activation=relu)(add_block_2)
    block_3_dropout = Dropout(0.3)(block_3_conv_1)
    block_3_conv_2 = Conv3D(filters * 4, kernel_size, padding= "same", activation=relu)(block_3_dropout)
    add_block_3 = Add()([block_3_conv_1, block_3_conv_2])

    #128 filters
    block_4_conv_1 = Conv3D(filters * 8, kernel_size, strides=(2, 2, 2), padding= "same", activation=relu)(add_block_3)
    block_4_dropout = Dropout(0.3)(block_4_conv_1)
    block_4_conv_2 = Conv3D(filters * 8, kernel_size, padding= "same", activation=relu)(block_4_dropout)
    add_block_4 = Add()([block_4_conv_1, block_4_conv_2])

    #256 filters
    block_5_conv_1 = Conv3D(filters * 16, kernel_size, strides=(2, 2, 2), padding= "same", activation=relu)(add_block_4)
    block_5_dropout = Dropout(0.3)(block_5_conv_1)
    block_5_conv_2 = Conv3D(filters * 16 , kernel_size, padding= "same", activation=relu)(block_5_dropout)
    add_block_5 = Add()([block_5_conv_1, block_5_conv_2])

    



if __name__ == "__main__":
   model()
    