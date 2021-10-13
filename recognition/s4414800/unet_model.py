"""
Model architecture for improved UNet.

@author Yin Peng
@email yin.peng@uqconnect.edu.au
"""

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential,Model

def get_improved_unet(input_shape):
  inputs = Input(input_shape)
  conv1_1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(inputs)
  conv1_2 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv1_1)
  # (None, 256, 128, 16)
  merg1 = Add()([conv1_1,conv1_2])

  # (None, 256, 128, 16)

  pool1 = MaxPooling2D(pool_size = (2, 2))(merg1)
  # (None, 128, 64, 16)
  
  conv2_1 = Conv2D(32, 3, activation = 'relu', padding = 'same')(pool1)
  conv2_2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv2_1)
  # (None, 128, 64, 32)

  merg2 = Add()([conv2_1,conv2_2])
  pool2 = MaxPooling2D(pool_size = (2, 2))(merg2)
  # (None, 64, 32, 32)

  conv3_1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(pool2)
  conv3_2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv3_1)
  # (None, 64, 32, 64)
  merg3 = Add()([conv3_1,conv3_2])

  pool3 = MaxPooling2D(pool_size = (2, 2))(merg3)
  # (None, 32, 16, 64)

  conv4_1 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool3)
  conv4_2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv4_1)

  # (None, 32, 16, 128)

  merg4 = Add()([conv4_1,conv4_2])

  drop4 = Dropout(0.3)(merg4)
  # (None, 32, 16, 128)
  pool4 = MaxPooling2D(pool_size = (2, 2))(drop4)

  # (None, 16, 8, 128)
  
  conv5_1 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool4)
  conv5_2 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv5_1)
  # (None, 16, 8, 256)
  merg5 = Add()([conv5_1,conv5_2])

  drop5 = Dropout(0.3)(merg5)

  up6 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2, 2))(drop5))
  # (None, 32, 16, 128)

  merge6 = concatenate([drop4, up6], axis = 3)
  # (None, 32, 16, 256)
  conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge6)
  # (None, 32, 16, 128)

  up7 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2, 2))(conv6))
  # (None, 64, 32, 64)
  merge7 = concatenate([merg3, up7], axis = 3)
  # (None, 64, 32, 128)
  conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge7)
  # (None, 64, 32, 64)

  up8 = Conv2D(32, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2, 2))(conv7))
  #(None, 128, 64, 32)
  merge8 = concatenate([merg2, up8], axis = 3)
  #(None, 128, 64, 64)
  conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same')(merge8)
  #(None, 128, 64, 32)

  up9 = Conv2D(16, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2, 2))(conv8))
  # (None, 256, 128, 16)
  merge9 = concatenate([merg1, up9], axis = 3)
  # (None, 256, 128, 32)
  conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same')(merge9)
  # (None, 256, 128, 32)
  
  conv_on_7 = Conv2D(32, 2, activation = 'relu', padding = 'same')(conv7)
  # (None, 64, 32, 32)
  add1 = Add()([UpSampling2D(size = (2, 2))(conv_on_7),conv8])
  # (None, 128, 64, 32)
  add2 = Add()([UpSampling2D(size = (2, 2))(add1),conv9])
  # (None, 256, 128, 32)

  conv10 = Conv2D(1, 1, activation = 'softmax')(add2)
  # (None, 256, 128, 1)

  model = Model(inputs = inputs, outputs = conv10)

  return model