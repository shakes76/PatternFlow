#Conv2DMod layer
#https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0/blob/989306792ca49dcbebb353c4f06c7b48aeb3a9e3/conv_mod.py#L15

#import stuff
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.layers import InputSpec

#define the layer
class ModConv2D (keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1,1), padding = 'valid', kernel_initializer='glorot_uniform', 
                    kernel_regularizer=None, activity_regularizer=None, kernel_constraint=None, demod = True, **kwargs):
        #define all the parameters of the layer
        super(ModConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.rank = 2
        self.kernel_size= conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        #?
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        #?
        self.demod = demod

        #input with ndim=4 is previous convolution layer
        #input with ndim=2 is the input style for this layer (output from style generator)
        self.Input_spec = [InputSpec(ndim=4), InputSpec(ndim=2)]


    def build(self, input_shape):
        #define weights after shape of input is known
        channel_axis = -1
        input_dim = input_shape[0][channel_axis] #should be 1 for this dataset since it's only grayscale images being sent through the system
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape, initializer=self.kernel_initializer, name='kernel', 
                                        regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)

        #input specifications
        #input_shape[0] is the output of the previous layer
        #input_shape[1] is the style
        self.input_spec = [InputSpec(ndim=4, axes={channel_axis: input_dim}), InputSpec(ndim=2)]

        self.built = True


    def call(self, inputs):
        #execute the code when the layer is used
            #modulation stuff
        style = inputs[1]
        #print("style shape:", style.shape)
        #print("kernel shape:", self.kernel.shape)

        #make the input style W shape compatible with kernel
        inp_mods = K.expand_dims(K.expand_dims(K.expand_dims(style, axis = 1), axis = 1), axis = -1)
        my_kernel = K.expand_dims(self.kernel, axis=0)

        #modulate
        #print("kernel", (int)(tf.rank(my_kernel)), my_kernel.shape)
        #print("kernel shape:", my_kernel.shape)
        #print("input style", (int)(tf.rank(inp_mods)), inp_mods.shape)
        #print("input style shape:", inp_mods.shape)
        weights = my_kernel * (inp_mods + 1)
        #weights = 0

        #demodulate
        if self.demod:
            weights /= K.sqrt(K.sum(K.square(weights), axis=[1,2,3], keepdims = True) + 1e-8)
        
        x = tf.transpose(inputs[0], [0,3,1,2])
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]])

        w = tf.transpose(weights, [1,2,3,0,4])
        w = tf.reshape(w, [weights.shape[1], weights.shape[2], weights.shape[3], -1])

        #normal convolution 2d
        #data is stored in [batch_size, channels, height, width]
        x = tf.nn.conv2d(x, w, strides=self.strides, padding='SAME', data_format='NCHW')

        #print(x.shape)

        x = tf.reshape(x, [-1, self.filters, tf.shape(x)[2], tf.shape(x)[3]]) # Fused => reshape convolution groups back to minibatch.
        x = tf.transpose(x, [0, 2, 3, 1])

        return x