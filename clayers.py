import os

import numpy as np
import tensorflow as tf
from keras import backend
from keras.initializers import RandomNormal
from keras.layers import Add, Conv2D, Dense, Layer, LeakyReLU
from PIL import Image

from config import *


# normalize weight by given shape
# https://prateekvishnu.medium.com/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
# stddev = sqrt(2 / fan_in)
class HeNormal(Layer):

    def __init__(self, shape, gain=2, **kwargs):
        super(HeNormal, self).__init__(**kwargs)
        shape = np.asarray(shape)
        shape = tf.constant(shape, dtype=tf.float32)
        # number of units, prod of dims
        fan_in = tf.math.reduce_prod(shape)
        self.scale = tf.math.sqrt(gain / fan_in)

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.float32)
        return inputs * self.scale


class Bias(Layer):

    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        zeros = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=zeros(shape=(input_shape[-1],), dtype='float32'), trainable=True)

    def call(self, inputs, **kwargs):
        return inputs + self.bias


# normalized dense layer
def EqualDense(x, filters, gain=1.):
    init = RandomNormal(stddev=1.)
    in_filters = backend.int_shape(x)[-1]
    x = Dense(filters, use_bias=False, kernel_initializer=init, dtype='float32')(x)
    x = HeNormal(shape=(in_filters), gain=gain)(x)
    x = Bias(input_shape=x.shape)(x)
    return x


# normalized conv layer
def EqualConv(x, filters, gain=2., kernel=(3, 3), strides=(1, 1)):
    init = RandomNormal(mean=0., stddev=1.)
    in_filters = backend.int_shape(x)[-1]
    x = Conv2D(filters, kernel, strides=strides, use_bias=False, padding="same", kernel_initializer=init, dtype='float32')(x)
    x = HeNormal(shape=(in_filters), gain=gain)(x)
    x = Bias(input_shape=x.shape)(x)
    return x


# adaptive instance normalization
class AdaIN(Layer):

    def __init__(self, **kwargs):
        super(AdaIN, self).__init__(**kwargs)

    def build(self, input_shapes):
        filters = input_shapes[0][-1]
        self.dense_ys = Dense(filters)
        self.dense_yb = Dense(filters)
        self.henorm_ys = HeNormal(shape=(filters))
        self.henorm_yb = HeNormal(shape=(filters))

    def call(self, inputs):
        x, w = inputs

        # instance normalization
        m, v = tf.nn.moments(x, [1, 2], keepdims=True)
        x = (x - m) / tf.sqrt(v + EPS)

        # w -> A
        ys = self.henorm_ys(self.dense_ys(w))
        yb = self.henorm_yb(self.dense_yb(w))

        # reshape for calculation
        ys = tf.reshape(ys, (-1, 1, 1, x.shape[-1]))
        yb = tf.reshape(yb, (-1, 1, 1, x.shape[-1]))

        return ys * x + yb


# add noise (B) to feature maps, after conv before adain
class AddNoise(Layer):

    def build(self, input_shape):
        init = RandomNormal(stddev=1.)
        filters = input_shape[0][-1]
        # without specifying 'name' excption raised when saving weights. bug?
        self.b = self.add_weight(shape=[1, 1, 1, filters], initializer=init, trainable=True, name='w')
        self.lrelu = LeakyReLU(0.2)

    def call(self, inputs):
        x, B = inputs
        x = x + self.b * B
        return self.lrelu(x)


# Mini Batch Standadization
# average stdd of all features and spatial location.
# concat after creating a constant feature map with the average stddev
class MinibatchStdev(Layer):

    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    def call(self, inputs):
        # mean accross channels
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        # std accross channels
        stddev = tf.sqrt(tf.reduce_mean(tf.square(inputs - mean), axis=0, keepdims=True) + EPS)
        average_stddev = tf.reduce_mean(stddev, keepdims=True)
        shape = tf.shape(inputs)
        minibatch_stddev = tf.tile(average_stddev, (shape[0], shape[1], shape[2], 1))
        combined = tf.concat([inputs, minibatch_stddev], axis=-1)
        return combined

    def compute_output_shape(self, input_shape):
        # one stats layer added
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)


# Weighted Sum
# alpha update during training
class WeightedSum(Add):

    def __init__(self, alpha=0., **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha)
        
    def call(self, inputs):
        a, b = inputs
        wsum = (1. - self.alpha) * a + self.alpha * b
        return wsum


# callback to update alpha during training
class FadeInCallBack(tf.keras.callbacks.Callback):

    def __init__(self):
        self.iters_per_epoch = 0 # number of iterations of each epoch
        self.epochs = 0          # total number of epochs
        self.iters = 0           # total number of iterations (iters per epoch * total epochs)
        self.current_epoch = 0   # current epoch

    def set_iters(self, epochs, iters_per_epoch):
        self.epochs = epochs
        self.steps_per_epoch = iters_per_epoch
        # total iters = epochs * steps_per_epoch
        self.iters = epochs * iters_per_epoch

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
    
    def set_alpha(self, alpha):
        self.alpha = alpha

    def on_batch_begin(self, current_iter, logs=None):
        # update alpha for fade-in layers
        # alpha = current iteration / total iterations
        alpha = ((self.current_epoch * self.steps_per_epoch) + current_iter + 1) / float(self.iters)

        # update alpha for both G and D
        for layer in self.model.G.layers + self.model.D.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)


# callback for generating images after each epoch
class SamplingCallBack(tf.keras.callbacks.Callback):

    def __init__(
        self,
        output_num_img = NSAMPLES,        # number of output images
        output_img_res = 256,             # output image resolution/size
        output_img_folder = IMAGE_DIR,    # output image folder
        output_ckpts_folder = CKPTS_DIR,  # checkpoints foler
        is_rgb = CHANNELS > 1,            # is output image rgb?
        seed = 3710                       # seed for z
    ):
        self.output_num_img = output_num_img          
        self.output_img_dim = output_img_res          
        self.output_img_mode = 'RGB' if is_rgb else 'L'
        self.output_img_folder = output_img_folder     
        self.output_ckpts_folder = output_ckpts_folder 
        self.seed = seed                               

    def set_prefix(self, prefix):
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        sgan = self.model
        
        # build inputs for G. constant, z, w, noise(B)
        const = tf.ones([self.output_num_img, SRES, SRES, FILTERS[0]])
        z = tf.random.normal((self.output_num_img, LDIM), seed=self.seed)
        ws = sgan.FC(z)
        inputs = [const]
        for i in range(sgan.current_depth+1):
            w = ws[:, i]
            B = tf.random.normal((self.output_num_img, SRES*(2**i), SRES*(2**i), 1))
            inputs += [w, B]

        # generate
        samples = sgan.G(inputs)

        # save
        w = h = int(np.sqrt(self.output_num_img))
        combined_image = Image.new(self.output_img_mode, (self.output_img_dim * w, self.output_img_dim * h))
        for i in range(self.output_num_img):
            image = tf.keras.preprocessing.image.array_to_img(samples[i])
            image = image.resize((self.output_img_dim, self.output_img_dim))
            combined_image.paste(image, (i % w * self.output_img_dim, i // h * self.output_img_dim))
        path = os.path.join(self.output_img_folder, f'{self.prefix}_{epoch+1:02d}.png')
        combined_image.save(path)
        print(f'\n{self.output_num_img} progress images saved: {path}')
