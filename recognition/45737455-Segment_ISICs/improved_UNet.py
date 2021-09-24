import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dropout, LeakyReLU, UpSampling2D, Softmax, concatenate
from tensorflow.python.keras.engine.input_layer import InputLayer

import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization
# from tensorflow.keras.activations import 


def improved_UNet(input_shape=(256, 256, 1), n_filters=16, dropout_rate=0.3, leaky_slope=1e-2):
    input_layer = InputLayer(input_shape=input_shape)

    # context pathway 1
    con1_conv = Conv2D(n_filters*1, kernel_size=3, strides=1, padding=1)(input_layer) # TODO: padding=same, use_bias
    con1_context = context_module(n_filters*1, n_filters*1, leaky_slope, dropout_rate)(con1_conv)
    con1_context += con1_conv # element-wise sum

    # context pathway 2
    con2_conv = Conv2D(n_filters*2, kernel_size=3, strides=2, padding=1)(con1_context)
    con2_context = context_module(n_filters*2, n_filters*2, leaky_slope, dropout_rate)(con2_conv)
    con2_context += con2_conv

    # context pathway 3
    con3_conv = Conv2D(n_filters*4, kernel_size=3, strides=2, padding=1)(con2_context)
    con3_context = context_module(n_filters*4, n_filters*4, leaky_slope, dropout_rate)(con3_conv)
    con3_context += con3_conv

    # context pathway 4
    con4_conv = Conv2D(n_filters*8, kernel_size=3, strides=2, padding=1)(con3_context)
    con4_context = context_module(n_filters*8, n_filters*8, leaky_slope, dropout_rate)(con4_conv)
    con4_context += con4_conv

    # context pathway 5 & localization pathway 1
    con5_conv = Conv2D(n_filters*16, kernel_size=3, strides=2, padding=1)(con4_context)
    con5_context = context_module(n_filters*16, n_filters*16, leaky_slope, dropout_rate)(con5_conv)
    con5_context += con5_conv

    # UP
    local1_up = UpSampling2D()(con5_context)

    # localization pathway 2
    local2_concat = concatenate([local1_up, con4_context])
    local2_localization = localization_module(n_filters*8, n_filters*8, leaky_slope)(local2_concat)
    local2_up = UpSampling2D()(local2_localization)
    # local2_conv = Conv2D() # TODO

    # localization pathway 3
    local3_concat = concatenate([local2_up, con3_context])
    local3_localization = localization_module(n_filters*4, n_filters*4, leaky_slope)(local3_concat)
    local3_up = UpSampling2D()(local3_localization)

    # localization pathway 4
    local4_concat = concatenate([local3_up, con2_context])
    local4_localization = localization_module(n_filters*2, n_filters*2, leaky_slope)(local4_concat)
    local4_up = UpSampling2D()(local4_localization)

    # last level
    last_conv = Conv2D(n_filters*8, 3, strides=1, padding=1)(local4_up)
    last_seg = Conv2D(n_filters*4, 3, strides=1, padding=1)(last_conv)
    # last_softmax = Softmax()(last_seg)

    outputs = last_seg + UpSampling2D(local4_localization + UpSampling2D(local3_localization))

    return Model(input = input_layer, output = outputs)


def context_module(input_shape, n_filters, leaky_slope=1e-2, dropout_rate=0.3):

    input_layer = InputLayer(input_shape=input_shape)

    # 1st
    norm1 = InstanceNormalization()(input_layer)
    leakyRelu1 = LeakyReLU(alpha=leaky_slope)(norm1)
    conv1 = Conv2D(filters=n_filters, kernel_size=3, strides=1, padding=1)(leakyRelu1)

    dropout = Dropout(dropout_rate)(conv1)

    # 2nd
    norm2 = InstanceNormalization()(dropout)
    leakyRelu2 = LeakyReLU(alpha=leaky_slope)(norm2)
    conv2 = Conv2D(filters=n_filters, kernel_size=3, strides=1, padding=1)(leakyRelu2)

    return conv2


def localization_module(input_shape, n_filters, leaky_slope=1e-2):

    input_layer = InputLayer(input_shape=input_shape)

    conv1 = Conv2D(filters=n_filters, kernel_size=3, strides=1, padding=1)(input_layer)
    # InstanceNormalization()
    leakyRelu1 = LeakyReLU(alpha=leaky_slope)(conv1)

    conv2 = Conv2D(filters=n_filters, kernel_size=1, strides=1, padding=0)(leakyRelu1)
    # InstanceNormalization()
    leakyRelu2 = LeakyReLU(alpha=leaky_slope)(conv2)

    return leakyRelu2

    
# class improved_UNet(Model):
#     def __init__(self, n_filters=16, dropout_rate=0.3, leaky_slope=1e-2): # inputshape?
#         super(improved_UNet, self).__init__()

#         # self.input_shape = input_shape
#         self.n_filters = n_filters
#         self.dropout_rate=dropout_rate
#         self.leaky_slope=leaky_slope

#         # self.input_layer = InputLayer(input_shape=self.input_shape) # TODO: batch_size ?

#         # context pathway 1
#         self.con1_conv = Conv2D(self.n_filters*1, kernel_size=3, strides=1, padding=1) # TODO: padding=same, use_bias
#         self.con1_context = self.context_module(self.n_filters*1, self.n_filters*1)

#         # context pathway 2
#         self.con2_conv = Conv2D(self.n_filters*2, kernel_size=3, strides=2, padding=1)
#         self.con2_context = self.context_module(self.n_filters*2, self.n_filters*2)

#         # context pathway 3
#         self.con3_conv = Conv2D(self.n_filters*4, kernel_size=3, strides=2, padding=1)
#         self.con3_context = self.context_module(self.n_filters*4, self.n_filters*4)

#         # context pathway 4
#         self.con4_conv = Conv2D(self.n_filters*8, kernel_size=3, strides=2, padding=1)
#         self.con4_context = self.context_module(self.n_filters*8, self.n_filters*8)

#         # context pathway 5 & localization pathway 1
#         self.con5_conv = Conv2D(self.n_filters*16, kernel_size=3, strides=2, padding=1)
#         self.con5_context = self.context_module(self.n_filters*16, self.n_filters*16)

#         self.local1_up = UpSampling2D()

#         # localization pathway 2
#         self.local2_localization = self.localization_module(self.n_filters*8, self.n_filters*8)
#         self.local2_up = UpSampling2D()

#         # localization pathway 3
#         self.local3_localization = self.localization_module(self.n_filters*4, self.n_filters*4)
#         self.local3_up = UpSampling2D()

#         # localization pathway 4
#         self.local4_localization = self.localization_module(self.n_filters*2, self.n_filters*2)
#         self.local4_up = UpSampling2D()

#         # last level
#         self.last_conv = Conv2D(filters=self.n_filters*8, kernel_size=3, strides=1, padding=1)
#         self.last_seg = Conv2D(filters=self.n_filters*4, kernel_size=3, strides=1, padding=1)
#         self.last_softmax = Softmax()        


#     def context_module(self, input_shape, n_filters):
#         module = Sequential(
#             [
#                 InputLayer(input_shape),

#                 # 1st
#                 InstanceNormalization(),
#                 LeakyReLU(alpha=self.leaky_slope),
#                 Conv2D(filters=n_filters, kernel_size=3, strides=1, padding=1),

#                 Dropout(self.dropout_rate),

#                 # 2nd
#                 InstanceNormalization(),
#                 LeakyReLU(alpha=self.leaky_slope),
#                 Conv2D(filters=n_filters, kernel_size=3, strides=1, padding=1),
#             ]
#         )

#         return module

#     def localization_module(self, input_shape, n_filters):
#         module = Sequential(
#             [
#                 InputLayer(layers=input_shape),

#                 Conv2D(filters=n_filters, kernel_size=3, strides=1, padding=1),
#                 # InstanceNormalization(),
#                 LeakyReLU(alpha=self.leaky_slope),

#                 Conv2D(filters=n_filters, kernel_size=1, strides=1, padding=0),
#                 # InstanceNormalization(),
#                 LeakyReLU(alpha=self.leaky_slope)
#             ]
#         )

#         return module


#     def call(self, inputs):
#         outputs = self.con1_conv(inputs)
#         residual_1 = outputs #
#         outputs = self.con1_context(outputs) + residual_1 # element-wise sum
#         context_1 = outputs # 

#         outputs = self.con2_conv(outputs)
#         residual_2 = outputs # 
#         outputs = self.con2_context(outputs) + residual_2
#         context_2 = outputs # 

#         outputs = self.con3_conv(outputs)
#         residual_3 = outputs #
#         outputs = self.con3_context(outputs) + residual_3
#         context_3 = outputs # 

#         outputs = self.con4_conv(outputs)
#         residual_4 = outputs #
#         outputs = self.con4_context(outputs) + residual_4
#         context_4 = outputs # 

#         outputs = self.con5_conv(outputs)
#         residual_5 = outputs #
#         outputs = self.con5_context(outputs) + residual_5

#         # UP
#         outputs = self.local1_up(outputs)

#         outputs = concatenate([outputs, context_4])#, axis=)
#         outputs = self.local2_localization(outputs)
#         outputs = self.local2_up(outputs)

#         outputs = concatenate([outputs, context_3])
#         outputs = self.local3_localization(outputs)
#         seg_3 = UpSampling2D(outputs)
#         outputs = self.local3_up(outputs)

#         outputs = concatenate([outputs, context_2])
#         outputs = self.local4_localization(outputs)
#         seg_4 = UpSampling2D(outputs) + seg_3
#         outputs = self.local4_up(outputs)

#         outputs = concatenate([outputs, context_1])
#         outputs = self.last_conv(outputs)
#         seg_last = self.last_seg(outputs) + UpSampling2D(seg_4)
#         outputs = self.last_softmax(seg_last)

#         return outputs

# # model = improved_UNet()
