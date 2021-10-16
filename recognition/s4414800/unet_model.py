"""
Model architecture for improved UNet.

Code reference: https://arxiv.org/pdf/1802.10508v1.pdf

@author Yin Peng
@email yin.peng@uqconnect.edu.au
"""

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential,Model

def get_improved_unet(input_shape):
  def context_module(input,size):
    """
    function to get context model
    """
    out = Conv2D(size, 3, activation = 'relu', padding = 'same')(out)
    out = Dropout(0.3)(out)
    out = tfa.layers.InstanceNormalization()(out)
    out = Conv2D(size, 3, activation = 'relu', padding = 'same')(out)
    return out

  def segmentation_layer(layer):
    """
    return segmentation layers
    """
    return tf.keras.layers.Conv2D(1, (1,1), activation = 'sigmoid')(layer)

  inputs = Input(input_shape)

  conv1_1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(inputs)
  conv1_2 = context_module(conv1_1,16)
  add1 = Add()([conv1_1,conv1_2])

  # pool1 = MaxPooling2D(pool_size = (2, 2))(merg1)

  conv2_1 = Conv2D(32, 3, activation = 'relu', padding = 'same', strides = 2)(add1)
  conv2_2 = context_module(conv2_1,32)
  add2 = Add()([conv2_1,conv2_2])
  # (None, 128, 96, 32)

  conv3_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', strides = 2)(add2)
  conv3_2 = context_module(conv3_1,64)
  add3 = Add()([conv3_1,conv3_2])

  conv4_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', strides = 2)(add3)
  conv4_2 = context_module(conv4_1,128)
  add4 = Add()([conv4_1,conv4_2])

  conv5_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', strides = 2)(add4)
  conv5_2 = context_module(conv5_1,256)
  add5 = Add()([conv5_1,conv5_2])
  # (None, 16, 12, 256)

  up1 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2, 2))(add5))

  merge1 = concatenate([add4, up1])
  up_conv1 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding ='same')(merge1)
  up_conv1 = tf.keras.layers.Conv2D(128, 1, activation = 'relu', padding ='same')(up_conv1)

  up2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2, 2))(up_conv1))
  merge2 = concatenate([add3, up2])
  up_conv2 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding ='same')(merge2)
  up_conv2 = tf.keras.layers.Conv2D(64, 1, activation = 'relu', padding ='same')(up_conv2)

  up3 = Conv2D(32, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2, 2))(up_conv2))
  merge3 = concatenate([add2, up3])
  up_conv3 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding ='same')(merge3)
  up_conv3 = tf.keras.layers.Conv2D(32, 1, activation = 'relu', padding ='same')(up_conv3)

  up4 = Conv2D(16, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2, 2))(up_conv3))
  merge4 = concatenate([add1, up4])
  up_conv4 = tf.keras.layers.Conv2D(16, 3, activation = 'relu', padding ='same')(merge4)

  seg1 = segmentation_layer(up_conv2)
  seg2 = segmentation_layer(up_conv3)

  add_seg1 = Add()([UpSampling2D(size = (2, 2))(seg1),seg2])

  seg4 = segmentation_layer(up_conv4)
  add_seg2 = Add()([UpSampling2D(size = (2, 2))(add_seg1),seg4])

  output = Conv2D(1, 1, activation = 'sigmoid')(add_seg2)

  model = Model(inputs = inputs, outputs = output)

  return model
