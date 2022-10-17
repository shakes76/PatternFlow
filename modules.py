import numpy as np
import tensorflow as tf
from keras.layers import Activation, AveragePooling2D, Flatten, Input, LeakyReLU, UpSampling2D
from keras.models import Model

import clayers as custom_layers
from config import *


# stylegan model
# used model in paper with less layers of FC and 1 color channel
# z -> w -> A -> （const, A, B) -> AdaIn -> conv 3x3 -> （A, B) AdaIn -> ...
# https://arxiv.org/abs/1812.04948
class StyleGAN(Model):

    def __init__(self):
        super(StyleGAN, self).__init__()
        self.DEPTH = int(np.log2(TRES) - np.log2(SRES))  # training depth
        self.current_depth = 0                           # current training depth
        self.FC = custom_layers.fc(self.DEPTH)           # FC net
        self.G = self.init_G()                           # generator
        self.D = self.init_D()                           # discriminator
    
    def compile(self, d_optimizer, g_optimizer):
        super(StyleGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        
    def init_D(self):
        image = Input(shape=(SRES, SRES, CHANNELS))
        x = custom_layers.EqualConv(image, out_filters=FILTERS[0], kernel=(1, 1))
        x = LeakyReLU(0.2)(x)
        x = custom_layers.MinibatchStd()(x)
        x = custom_layers.EqualConv(x, out_filters=FILTERS[0])
        x = LeakyReLU(0.2)(x)
        x = custom_layers.EqualConv(x, out_filters=FILTERS[0], kernel=(4, 4), strides=(4, 4))
        x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = custom_layers.EqualDense(x, out_filters=CHANNELS)
        return Model(image, x)

    # grow discriminator
    def grow_D(self):
        input_shape = list(self.D.input.shape)
        
        # w*2, h*2, c 
        input_shape = (input_shape[1]*2, input_shape[2]*2, input_shape[3])
        image = Input(shape=input_shape)
        image = tf.cast(image, tf.float32)

        # downsample
        x1 = AveragePooling2D()(image)
        for i in [1, 2, 3, 4]:
            x1 = self.D.layers[i](x1)

        f = FILTERS[self.current_depth]
        
        x2 = custom_layers.EqualConv(image, out_filters=f, kernel=(1, 1))
        x2 = LeakyReLU(0.2)(x2)
        x2 = custom_layers.EqualConv(x2, out_filters=f)
        x2 = LeakyReLU(0.2)(x2)
        x2 = custom_layers.EqualConv(x2, out_filters=FILTERS[self.current_depth-1])
        x2 = LeakyReLU(0.2)(x2)
        x2 = AveragePooling2D()(x2)
        x = custom_layers.WeightedSum()([x1, x2])

        for i in range(5, len(self.D.layers)):
            x2 = self.D.layers[i](x2)
        self.D_ST = Model(image, x2)

        for i in range(5, len(self.D.layers)):
            x = self.D.layers[i](x)
        self.D = Model(image, x)

    def init_G(self):
        # initialize generator
        # base block: 3 inputs, constant, w, noise(B)
        
        r = SRES
        f = FILTERS[0]
        
        const = Input(shape=(r, r, f), name='Constant')
        w = Input(shape=(LDIM), name='w(0)')
        B = Input(shape=(r, r, 1), name='B(0)')
        x = const
        
        x = custom_layers.AddNoise()([x, B])
        x = LeakyReLU(0.2)(x)
        x = custom_layers.AdaIN()([x, w])
        x = custom_layers.EqualConv(x, out_filters=f)
        x = LeakyReLU(0.2)(x)
        
        x = custom_layers.AddNoise()([x, B])
        x = LeakyReLU(0.2)(x)
        x = custom_layers.AdaIN()([x, w])
        x = custom_layers.EqualConv(x, out_filters=CHANNELS, kernel=(1, 1), gain=1.)
        x = Activation('tanh', name='tanh_0')(x)
        
        return Model([const, w, B], x)

    # grow generator (fade in)
    def grow_G(self):
        d = self.current_depth
        f = FILTERS[d]
        res = SRES*(2**d) 
        
        # extract, expand end of torgb
        end = self.G.layers[-5].output
        end = UpSampling2D((2, 2))(end)

        # branch
        x1 = end
        for i in [-4, -3, -2, -1]:
            x1 = self.G.layers[i](x1)

        # branch
        w = Input(shape=(LDIM), name=f'w({d})')
        B = Input(shape=(res, res, 1), name=f'B({d})')
        
        x2 = custom_layers.EqualConv(end, out_filters=f)
        x2 = LeakyReLU(0.2)(x2)
        x2 = custom_layers.AddNoise()([x2, B])
        x2 = LeakyReLU(0.2)(x2)
        x2 = custom_layers.AdaIN()([x2, w])
        
        x2 = custom_layers.EqualConv(x2, out_filters=f)
        x2 = LeakyReLU(0.2)(x2)
        x2 = custom_layers.AddNoise()([x2, B])
        x2 = LeakyReLU(0.2)(x2)
        x2 = custom_layers.AdaIN()([x2, w])
        
        # to rgb
        x2 = custom_layers.EqualConv(x2, out_filters=CHANNELS, kernel=(1, 1), gain=1.)
        x2 = Activation('tanh', name=f'tanh_{d}')(x2)

        # stabilize
        self.G_ST = Model(self.G.input+[w,B], x2)
        
        # fade in
        self.G = Model(self.G.input+[w,B], custom_layers.WeightedSum()([x1, x2]))

    # gradient constraint, to enforece unit norm gradient.
    # E[(grad(f(x))-1)^2]
    def gradient_penalty(self, batch_size, real_images, fake_images):
        # interpolated image
        w = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = (1 - w) * real_images + w * fake_images

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.D(interpolated, training=True)

        # gradient w.r.t to interpolated image
        grads = tape.gradient(pred, [interpolated])[0]
        # norm of the gradient
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        
        return gp
    
    def grow(self):
        self.current_depth += 1
        self.grow_G()
        self.grow_D()

    def stabilize(self):
        self.G = self.G_ST
        self.D = self.D_ST

    # customized train step
    def train_step(self, data):
        real_images = data[0]
        batch_size = tf.shape(real_images)[0]
        const = tf.ones([batch_size, SRES, SRES, FILTERS[0]])
        
        # train discriminator
        with tf.GradientTape() as tape:
            
            # build input for G: [const, w, B, w, B, w, B, ...]
            z = tf.random.normal(shape=(batch_size, LDIM))
            ws = self.FC(z)
            inputs = [const]
            for i in range(self.current_depth+1):
                w = ws[:, i]
                B = tf.random.normal((batch_size, SRES*(2**i), SRES*(2**i), 1))
                inputs += [w, B]
            
            # generate fake images
            fake_images = self.G(inputs, training=True)
            fake_pred = self.D(fake_images, training=True)
            real_pred = self.D(real_images, training=True)
            # wasserstein
            d_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)

            # gradient penalty, lambda 10
            penulty = 10 * self.gradient_penalty(batch_size, real_images, fake_images)
            
            # drift for regularization, drift weight 0.001
            drift = .001 * tf.reduce_mean(tf.square(real_pred))
            
            # discriminator loss = original discriminator loss + penulty + drift
            # lambda=10, drift weight = 0.001
            d_loss = d_loss + penulty + drift

        d_grad = tape.gradient(d_loss, self.D.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_grad, self.D.trainable_weights))

        # train generator
        with tf.GradientTape() as tape:
            z = tf.random.normal(shape=(batch_size, LDIM))
            ws = self.FC(z)
            inputs = [const]
            for i in range(self.current_depth+1):
                w = ws[:,i]
                B = tf.random.normal((batch_size, SRES*(2**i), SRES*(2**i), 1))
                inputs += [w, B]
                
            fake_images = self.G(inputs, training=True)
            fake_pred = self.D(fake_images, training=True)
            
            # wasserstein
            g_loss = -tf.reduce_mean(fake_pred)
            
        # grad w.r.t fc layers and generator
        trainable_weights = self.FC.trainable_weights + self.G.trainable_weights
        g_grad = tape.gradient(g_loss, trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grad, trainable_weights))
        
        return {'d_loss': d_loss, 'g_loss': g_loss}
