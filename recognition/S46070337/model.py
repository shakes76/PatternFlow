from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow as tf
import matplotlib.pyplot as plt


# Build module function
def block(input_data, conv_size):
    conv1 = tfa.layers.InstanceNormalization()(input_data)
    conv2 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv1)
    conv3 = tf.keras.layers.Conv2D(conv_size, kernel_size=3, padding='same')(conv2)
    conv4 = tf.keras.layers.Dropout(0.3)(conv3)
    conv5 = tfa.layers.InstanceNormalization()(conv4)
    conv6 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv5)
    conv7 = tf.keras.layers.Conv2D(conv_size, kernel_size=3, padding='same')(conv6)
    return conv7


# Add segmentation layers
def segmentation_layer(input_data):
    seg_layers = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(input_data)
    return seg_layers


def unet_model(image_height, image_width, image_channels):
    input_data = tf.keras.layers.Input((image_height, image_width, image_channels))
    function_for_activating = tf.keras.layers.LeakyReLU(alpha=0.01)
    # Encoding Blocks
    conv1_1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(input_data)
    conv1_2 = block(conv1_1, 16)
    add1 = layers.Add()([conv1_2, conv1_1])
    
    conv2_1 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same', strides=2)(add1)
    conv2_2 = block(conv2_1, 32)
    add2 = layers.Add()([conv2_2, conv2_1])
    
    conv3_1 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same', strides=2)(add2)
    conv3_2 = block(conv3_1, 64)
    add3 = layers.Add()([conv3_2, conv3_1])
    
    conv4_1 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same', strides=2)(
        add3)
    conv4_2 = block(conv4_1, 128)
    add4 = layers.Add()([conv4_2, conv4_1])
    
    conv5_1 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same', strides=2)(
        add4)
    conv5_2 = block(conv5_1, 256)
    add5 = layers.Add()([conv5_2, conv5_1])

    # Decoding and Upsampling
    up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(add5)
    up1_conv1 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(up1)
    up1_conv2 = tf.keras.layers.concatenate([add4, up1_conv1])
    up1_conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(up1_conv2)
    up1_conv4 = tf.keras.layers.Conv2D(128, (1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(up1_conv3)
    
    up2 = tf.keras.layers.UpSampling2D(size=(2, 2))(up1_conv4)
    up2_conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(up2)
    up2_conv2 = tf.keras.layers.concatenate([add3, up2_conv1])
    up2_conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(up2_conv2)
    up2_conv4 = tf.keras.layers.Conv2D(64, (1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(up2_conv3)
    
    segmentation_slice1 = segmentation_layer(up2_conv4)
    segmentation_slice2 = tf.keras.layers.UpSampling2D(size=(2, 2))(segmentation_slice1)
    
    up3 = tf.keras.layers.UpSampling2D(size=(2, 2))(up2_conv4)
    up3_conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(up3)
    up3_conv2 = tf.keras.layers.concatenate([add2, up3_conv1])
    up3_conv3 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(up3_conv2)
    up3_conv4 = tf.keras.layers.Conv2D(32, (1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(up3_conv3)
    
    segmentation_slice3 = segmentation_layer(up3_conv4)
    
    add6 = layers.Add()([segmentation_slice2, segmentation_slice3])
    conv6 = tf.keras.layers.UpSampling2D(size=(2, 2))(add6)
    
    up4 = tf.keras.layers.UpSampling2D(size=(2, 2))(up3_conv4)
    up4_conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(up4)
    up4_conv2 = tf.keras.layers.concatenate([add1, up4_conv1])
    up4_conv3 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(up4_conv2)
    
    segmentation_slice4 = segmentation_layer(up4_conv3)
    add7 = layers.Add()([conv6, segmentation_slice4])
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(add7)
    unetmodel = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    unetmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=["binary_crossentropy"],
                  metrics=["accuracy"])
    unetmodel.summary()
    
    return unetmodel
