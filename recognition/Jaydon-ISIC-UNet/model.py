'''
    model.py
    Author: Jaydon Hansen
    Date created: 26/10/2020
    Date last modified: 7/11/2020
    Python Version: 3.8
'''

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input

from layers import downsample, bottleneck, upsample


def UNet(img_size):
    """
    Generates a new UNet model with 4 downsampling layers, bottleneck and 4 upsampling layers
    """
    input = Input((img_size, img_size, 3))  # start at 128
    # build 4 downsampling layers
    ds1, pool = downsample(input, 16)  # downsample to 64
    ds2, pool = downsample(pool, 32)  # downsapmle to 32
    ds3, pool = downsample(pool, 64)  # downsample to 16
    ds4, pool = downsample(pool, 128)  # downsample to 8

    # add bottleneck
    bn = bottleneck(pool, 256)

    # add 4 upsampling layers
    us = upsample(bn, ds4, 128)  # upsample to 16
    us = upsample(us, ds3, 64)  # upsample to 32
    us = upsample(us, ds2, 32)  # upsample to 64
    us = upsample(us, ds1, 16)  # upsample to 128

    # output layer
    output = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(us)
    model = Model(input, output)
    return model
