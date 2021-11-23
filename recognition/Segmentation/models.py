'''
Models used for image segmentation. Contains implementation of improved UNET models from https://arxiv.org/abs/1802.10508v1
'''
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, UpSampling2D, Input, concatenate, LeakyReLU, Add
from tensorflow.keras import Model

# context module for improved unet
def _context_module(input_layer, filters):
    conv1 = Conv2D(filters, (3,3), padding='same',activation=LeakyReLU(0.01))(input_layer)
    dropout = Dropout(0.3)(conv1)
    conv2 = Conv2D(filters, (3,3), padding='same',activation=LeakyReLU(0.01))(dropout)
    return conv2

# upsampling module for improved unet
def _upsampling_module(input_layer, filters):
    upsample = UpSampling2D(size=(2,2))(input_layer)
    conv = Conv2D(filters, (3,3), padding='same',activation=LeakyReLU(0.01))(upsample)
    return conv

# localization module for improved unet
def _localization_module(input_layer, filters):
    conv1 = Conv2D(filters, (3,3), padding='same',activation=LeakyReLU(0.01))(input_layer)
    conv2 = Conv2D(filters, (1,1), padding='same',activation=LeakyReLU(0.01))(conv1)
    return conv2
    
def improved_unet(width, height, channels, output_classes, batch_size):
    inputs = Input((width, height, channels), batch_size)
    # inputs = Input((width, height, channels))
    # Level1 - Left
    conv1 = Conv2D(16, (3,3), padding='same',activation=LeakyReLU(0.01))(inputs)
    context_module1 = _context_module(conv1, 16)
    conv_layer1 = Add()([conv1, context_module1])

    # Level2 - Left
    conv2 = Conv2D(32, (3,3), 2, padding='same',activation=LeakyReLU(0.01))(conv_layer1)
    context_module2 = _context_module(conv2, 32)
    conv_layer2 = Add()([conv2, context_module2])

    # Level3 - Left
    conv3 = Conv2D(64, (3,3), 2, padding='same',activation=LeakyReLU(0.01))(conv_layer2)
    context_module3 = _context_module(conv3, 64)
    conv_layer3 = Add()([conv3, context_module3])

    # Level4 - Left
    conv4 = Conv2D(128, (3,3), 2, padding='same',activation=LeakyReLU(0.01))(conv_layer3)
    context_module4 = _context_module(conv4, 128)
    conv_layer4 = Add()([conv4, context_module4])

    # Level5
    conv5 = Conv2D(256, (3,3), 2, padding='same',activation=LeakyReLU(0.01))(conv_layer4)
    context_module5 = _context_module(conv5, 256)
    conv_layer5 = Add()([conv5, context_module5])
    upsampling_module5 = _upsampling_module(conv_layer5, 128)

    # Level4 - Right
    conc4 = concatenate([upsampling_module5, conv_layer4])
    local_module4 = _localization_module(conc4, 128)
    upsampling_module4 = _upsampling_module(local_module4, 64)

    # Level3 - Right
    conc3 = concatenate([upsampling_module4, conv_layer3])
    local_module3 = _localization_module(conc3, 64)
    upsampling_module3 = _upsampling_module(local_module3, 32)
    #Segmentation layer returns the segmented image with same channels as input image
    seg3 = Conv2D(channels, (1, 1), padding='same',activation=LeakyReLU(0.01))(local_module3)

    # Level2 - Right
    conc2 = concatenate([upsampling_module3, conv_layer2])
    local_module2 = _localization_module(conc2, 32)
    upsampling_module2 = _upsampling_module(local_module2, 16)

    # Segmentation layer between level 2 and 3
    seg2 = Conv2D(channels, (1, 1), padding='same',activation=LeakyReLU(0.01))(local_module2)
    up_seg3 = UpSampling2D(size=(2, 2))(seg3)
    add_seg2_3 = Add()([seg2, up_seg3])

    # Level1 - Right
    conc2 = concatenate([upsampling_module2, conv_layer1])
    conv1_1 = Conv2D(32, (3,3), padding='same',activation=LeakyReLU(0.01))(conc2)
    seg1 = Conv2D(channels, (1, 1), padding='same',activation=LeakyReLU(0.01))(conv1_1)
    up_seg2_3 = UpSampling2D(size=(2, 2))(add_seg2_3)
    add_seg1_2_3 = Add()([seg1, up_seg2_3])

    # Softmax
    outputs = Conv2D(output_classes, (1, 1), activation = 'softmax')(add_seg1_2_3)

    return Model(inputs=inputs, outputs=outputs)
