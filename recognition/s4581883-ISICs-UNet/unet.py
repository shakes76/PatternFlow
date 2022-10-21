import numpy as np
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, Dropout, BatchNormalization, Input, Add, concatenate, Activation
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

'''
File containing the UNet model and all related modules.
'''

# Call for each downsampling block in the model
def down_sampling(init, filter, strides, kernel_size = (3,3), padding = "same"):
    conv = Conv2D(filter, kernel_size, padding = padding, strides = strides) (init)
    conv = BatchNormalization() (conv)
    conv = LeakyReLU(0.01) (conv)
    conv = Dropout(0.3) (conv)
    conv = Conv2D(filter, kernel_size, padding = padding, strides = strides) (conv)
    conv = BatchNormalization() (conv)
    conv = LeakyReLU(0.01) (conv)
    return conv 

# Call for each upsampling block in the model
def upsampling(init, filters, kernel_size = (3,3), padding = "same", strides = 1):
    upsample = UpSampling2D((2,2)) (init)
    upsample = Conv2D(filters, kernel_size, padding, strides = strides) (upsample)
    upsample = BatchNormalization() (upsample)
    upsample = LeakyReLU(0.01) (upsample)
    return upsample

# Call for each localisation block in the model
def localize(init, filters, kernel_size = (3,3), padding = "same", strides = 1):
    local = Conv2D(filters, kernel_size, padding, strides = strides) (init)
    local = BatchNormalization() (local)
    local = LeakyReLU(0.01) (local)
    local = Conv2D(filters, kernel_size = (1,1), padding = padding) (init)
    local = BatchNormalization() (local)
    local = LeakyReLU(0.01) (local)
    return local

# Overall definition of the UNet model being implemented
def unet(height, width, channels, filters = 16, kernel_size = (3,3), padding = "same"):
    input = Input(shape = (width, height, channels))

    # ds1 - ds5 are the downsampling blocks as shown in the paper
    ds1 = Conv2D(filters, kernel_size = kernel_size, padding = padding) (input)
    ds1 = LeakyReLU(0.01) (ds1)
    ds1_call = down_sampling(ds1, filters, 1)
    ds1_output = Add()([ds1, ds1_call])

    ds2 = Conv2D(filters * 2, kernel_size = kernel_size, padding = padding) (ds1_output)
    ds2 = LeakyReLU(0.01) (ds2)
    ds2_call = down_sampling(ds2, filters * 2, 2)
    ds2_output = Add()([ds2, ds2_call])

    ds3 = Conv2D(filters * 4, kernel_size = kernel_size, padding = padding) (ds2_output)
    ds3 = LeakyReLU(0.01) (ds3)
    ds3_call = down_sampling(ds3, filters * 4, 2)
    ds3_output = Add()([ds3, ds3_call])
    
    ds4 = Conv2D(filters * 8, kernel_size = kernel_size, padding = padding) (ds3_output)
    ds4 = LeakyReLU(0.01) (ds4)
    ds4_call = down_sampling(ds4, filters * 8, 2)
    ds4_output = Add()([ds4, ds4_call])
    
    ds5 = Conv2D(filters * 16, kernel_size = kernel_size, padding = padding) (ds4_output)
    ds5 = LeakyReLU(0.01) (ds5)
    ds5_call = down_sampling(ds5, filters * 16, 2)
    ds5_output = Add()([ds5, ds5_call])

    # us1 - us4 are the upsampling blocks as shown in the paper
    us1 = upsampling(ds5_output, filters * 8)
    us1_output = Add()([ds4_output, us1])

    us2_l = localize(us1_output, filters * 8)
    us2 = upsampling(us2_l, filters * 4)
    us2_output = concatenate([ds3_output, us2])

    us3_l = localize(us2_output, filters * 4)
    us3 = upsampling(us3_l, filters * 2)
    us3_output = concatenate([ds2_output, us3])
    
    us4_l = localize(us3_output, filters * 2)
    us4 = upsampling(us4_l, filters)
    us4_output = concatenate([ds1_output, us4])
    
    # Segmentation layers 
    seg1 = Activation("sigmoid") (us2_l)
    seg1 = UpSampling2D(size=(8,8)) (seg1)
    seg2 = Activation("sigmoid") (us3_l)
    seg2 = UpSampling2D(size=(4,4))(seg2)
    seg3 = Conv2D(1, kernel_size = kernel_size, padding = padding, activation = "sigmoid")(us4_output)

    # Finalise the output
    output = Add()([seg1, seg2, seg3])
    output = Activation("softmax") (output)

    # Create the model
    unet_model = keras.models.Model(input, output)

    return unet_model