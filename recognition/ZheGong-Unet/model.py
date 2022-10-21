import tensorflow as tf
from tensorflow import keras

def unet_model(f,channel):

  # Block1
  inputs = keras.Input(shape = (256,256,1))
  conv1 = tf.keras.layers.Conv2D(4*f, (3, 3), padding = 'same')(inputs)
  conv1 = tf.keras.layers.LeakyReLU(0.01)(conv1)
  conv2 = tf.keras.layers.Conv2D(4*f, (3, 3), padding = 'same')(conv1)
  conv2 = tf.keras.layers.LeakyReLU(0.01)(conv2)
  conv2 = tf.keras.layers.Dropout(0.3)(conv2)
  conv2 = tf.keras.layers.Conv2D(4*f, (3, 3), padding = 'same')(conv2) 
  conv2 = tf.keras.layers.LeakyReLU(0.01)(conv2)
  conv2 = conv2 + conv1

  # Block2
  conv3 = tf.keras.layers.Conv2D(8*f, (3, 3), padding = 'same', strides = (2, 2))(conv2)
  conv3 = tf.keras.layers.LeakyReLU(0.01)(conv3)
  conv4 = tf.keras.layers.Conv2D(8*f, (3, 3), padding = 'same',)(conv3)
  conv4 = tf.keras.layers.LeakyReLU(0.01)(conv4)
  conv4 = tf.keras.layers.Dropout(0.3)(conv4)
  conv4 = tf.keras.layers.Conv2D(8*f, (3, 3), padding = 'same',)(conv4)
  conv4 = tf.keras.layers.LeakyReLU(0.01)(conv4)
  conv4 = conv4 + conv3

  # Block3
  conv5 = tf.keras.layers.Conv2D(16*f, (3, 3), padding =  'same', strides = (2, 2))(conv4)
  conv5 = tf.keras.layers.LeakyReLU(0.01)(conv5)
  conv6 = tf.keras.layers.Conv2D(16*f, (3, 3), padding = 'same')(conv5)
  conv6 = tf.keras.layers.LeakyReLU(0.01)(conv6)
  conv6 = tf.keras.layers.Dropout(0.3)(conv6)
  conv6 = tf.keras.layers.Conv2D(16*f, (3, 3), padding = 'same')(conv6)
  conv6 = tf.keras.layers.LeakyReLU(0.01)(conv6)
  conv6 = conv6 + conv5

  # Block4
  conv7 = tf.keras.layers.Conv2D(32*f, (3, 3), padding = 'same', strides = (2, 2))(conv6)
  conv7 = tf.keras.layers.LeakyReLU(0.01)(conv7)
  conv8 = tf.keras.layers.Conv2D(32*f, (3, 3), padding = 'same')(conv7)
  conv8 = tf.keras.layers.LeakyReLU(0.01)(conv8)
  conv8 = tf.keras.layers.Dropout(0.3)(conv8)
  conv8 = tf.keras.layers.Conv2D(32*f, (3, 3), padding = 'same')(conv8)
  conv8 = tf.keras.layers.LeakyReLU(0.01)(conv8)
  conv8 = conv8 + conv7

  # Block5
  conv9 = tf.keras.layers.Conv2D(64*f, (3, 3), padding = 'same', strides = (2, 2))(conv8)
  conv9 = tf.keras.layers.LeakyReLU(0.01)(conv9)
  conv10 = tf.keras.layers.Conv2D(64*f, (3, 3), padding = 'same')(conv9)
  conv10 = tf.keras.layers.LeakyReLU(0.01)(conv10)
  conv10 = tf.keras.layers.Dropout(0.3)(conv10)
  conv10 = tf.keras.layers.Conv2D(64*f, (3, 3), padding = 'same')(conv10)
  conv10 = tf.keras.layers.LeakyReLU(0.01)(conv10)
  conv10 = conv10 + conv9

  # Upssampling Block1
  up1 = tf.keras.layers.UpSampling2D(size = (2, 2))(conv10)
  up1 = tf.keras.layers.Conv2D(32*f, (2, 2), padding = 'same')(up1)
  up1 = tf.keras.layers.LeakyReLU(0.01)(up1)
  up1 = tf.concat([conv8,up1], axis = 3)

  # Upssampling Block2
  up2 = tf.keras.layers.Conv2D(32*f, (3, 3), padding = 'same')(up1)
  up2 = tf.keras.layers.LeakyReLU(0.01)(up2)
  up2 = tf.keras.layers.Conv2D(32*f, (1, 1), padding = 'same')(up2)
  up2 = tf.keras.layers.LeakyReLU(0.01)(up2)
  up2 = tf.keras.layers.UpSampling2D(size = (2, 2))(up2)
  up2 = tf.keras.layers.Conv2D(16*f, (2, 2), padding = 'same')(up2)
  up2 = tf.keras.layers.LeakyReLU(0.01)(up2)
  up2 = tf.concat([conv6,up2], axis = 3)

  # Upssampling Block3
  up3 = tf.keras.layers.Conv2D(16*f, (3, 3), padding = 'same')(up2)
  up3 = tf.keras.layers.LeakyReLU(0.01)(up3)
  up3_ = tf.keras.layers.Conv2D(16*f, (1, 1), padding = 'same')(up3)
  up3_ = tf.keras.layers.LeakyReLU(0.01)(up3_)
  up3 = tf.keras.layers.UpSampling2D(size = (2, 2))(up3_)
  up3 = tf.keras.layers.Conv2D(8*f, (2, 2), padding = 'same')(up3)
  up3 = tf.keras.layers.LeakyReLU(0.01)(up3)
  up3 = tf.concat([conv4,up3], axis = 3)

  # Upssampling Block4
  up4 = tf.keras.layers.Conv2D(8*f, (3, 3), padding = 'same')(up3)
  up4 = tf.keras.layers.LeakyReLU(0.01)(up4)
  up4_ = tf.keras.layers.Conv2D(8*f, (1, 1), padding = 'same')(up4)
  up4_ = tf.keras.layers.LeakyReLU(0.01)(up4_)
  up4 = tf.keras.layers.UpSampling2D(size = (2, 2))(up4_)
  up4 = tf.keras.layers.Conv2D(4*f, (2, 2), padding = 'same')(up4)
  up4 = tf.keras.layers.LeakyReLU(0.01)(up4)
  up4 = tf.concat([conv2,up4], axis = 3)

  # seg1
  conv = tf.keras.layers.Conv2D(8*f, (3, 3), padding = 'same')(up4)
  conv = tf.keras.layers.LeakyReLU(0.01)(conv)
  conv = tf.keras.layers.Conv2D(channel, (1, 1), padding = 'same')(conv)
  conv = tf.keras.layers.LeakyReLU(0.01)(conv)
  
  # seg2
  up4_ = tf.keras.layers.Conv2D(channel, (1, 1), padding = 'same')(up4_)
  up4_ = tf.keras.layers.LeakyReLU(0.01)(up4_)

  # seg3
  up3_ = tf.keras.layers.Conv2D(channel, (1, 1), padding = 'same')(up3_)
  up3_ = tf.keras.layers.LeakyReLU(0.01)(up3_)

  # element-wise sum of seg1, seg2 and seg3
  up3_ = tf.keras.layers.UpSampling2D(size = (2, 2))(up3_)
  up4_ = up3_ + up4_
  up4_ = tf.keras.layers.UpSampling2D(size = (2, 2))(up4_)
  conv = conv + up4_

  output = tf.keras.layers.Conv2D(channel, (1, 1), activation = 'softmax')(conv)
  model = keras.Model(inputs = inputs,outputs = output)
  return model