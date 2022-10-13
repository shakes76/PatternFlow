import numpy as np
import tensorflow as tf
from keras.layers import Activation, AveragePooling2D, Flatten, Input, UpSampling2D
from keras.models import Model

from clayers import *
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
        self.FC = self.mapping()                         # FC net
        self.G = self.init_G()                           # generator
        self.D = self.init_D()                           # discriminator
    
    def compile(self, d_optimizer, g_optimizer):
        super(StyleGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        
    # fully connected layers. z->w.
    def mapping(self):
        z = Input(shape=(LDIM))
        # 8 layers in paper. use 6 instead.
        w = EqualDense(z, out_filters=LDIM)
        w = LeakyReLU(0.2)(w)
        for _ in range(self.DEPTH-1):
            w = EqualDense(w, out_filters=LDIM)
            w = LeakyReLU(0.2)(w)
        # replicate (256,7)
        w = tf.tile(tf.expand_dims(w, 1), (1, self.DEPTH+1, 1)) 
        return Model(z, w)

    def init_D(self):
        image = Input(shape=(SRES, SRES, CHANNELS))
        x = EqualConv(image, out_filters=FILTERS[0], kernel=(1, 1))
        x = LeakyReLU(0.2)(x)
        x = MinibatchStd()(x)
        x = EqualConv(x, out_filters=FILTERS[0])
        x = LeakyReLU(0.2)(x)
        x = EqualConv(x, out_filters=FILTERS[0], kernel=(4, 4), strides=(4, 4))
        x = LeakyReLU(0.2)(x)
        x = EqualDense(Flatten()(x), out_filters=CHANNELS)
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
        
        x2 = EqualConv(image, out_filters=f, kernel=(1, 1))
        x2 = LeakyReLU(0.2)(x2)
        x2 = EqualConv(x2, out_filters=f)
        x2 = LeakyReLU(0.2)(x2)
        x2 = EqualConv(x2, out_filters=FILTERS[self.current_depth-1])
        x2 = LeakyReLU(0.2)(x2)
        x2 = AveragePooling2D()(x2)
        x = WeightedSum()([x1, x2])

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
        
        x = AddNoise()([x, B])
        x = LeakyReLU(0.2)(x)
        x = AdaIN()([x, w])
        x = EqualConv(x, out_filters=f)
        x = LeakyReLU(0.2)(x)
        
        x = AddNoise()([x, B])
        x = LeakyReLU(0.2)(x)
        x = AdaIN()([x, w])
        x = EqualConv(x, out_filters=CHANNELS, kernel=(1, 1), gain=1.)
        x = Activation('tanh')(x)
        
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
        
        x2 = EqualConv(end, out_filters=f)
        x2 = LeakyReLU(0.2)(x2)
        x2 = AddNoise()([x2, B])
        x2 = LeakyReLU(0.2)(x2)
        x2 = AdaIN()([x2, w])
        
        x2 = EqualConv(end, out_filters=f)
        x2 = LeakyReLU(0.2)(x2)
        x2 = AddNoise()([x2, B])
        x2 = LeakyReLU(0.2)(x2)
        x2 = AdaIN()([x2, w])
        
        # to rgb
        x2 = EqualConv(x2, out_filters=CHANNELS, kernel=(1, 1), gain=1.)
        x2 = Activation('tanh', name=f'tanh_{d}')(x2)

        # stabilize
        self.G_ST = Model(self.G.input+[w,B], x2)
        
        # fade in
        self.G = Model(self.G.input+[w,B], WeightedSum()([x1, x2]))

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

        
        fake_labels = -tf.ones(batch_size)
        real_labels = tf.ones(batch_size)
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
            
            # fake loss
            pred_fake_labels = self.D(fake_images, training=True)
            fake_loss = -tf.reduce_mean(fake_labels * pred_fake_labels)
            
            # real loss
            pred_real_labels = self.D(real_images, training=True)
            real_loss = -tf.reduce_mean(real_labels * pred_real_labels)
            
            # gradient penalty, lambda 10
            penulty = 10 * self.gradient_penalty(batch_size, real_images, fake_images)
            
            # drift for regularization, drift weight 0.001
            pred = tf.concat([pred_fake_labels, pred_real_labels], axis=0)
            drift = .001 * tf.reduce_mean(pred ** 2)
            
            # discriminator loss = real loss + fake loss + penulty + drift
            # lambda=10, drift weight = 0.001
            d_lost = fake_loss + real_loss + penulty + drift

        d_grad = tape.gradient(d_lost, self.D.trainable_weights)
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
            pred_fake_labels = self.D(fake_images, training=True)
            
            # D(G(z))
            g_loss = -tf.reduce_mean(real_labels * pred_fake_labels)
            
        # grad w.r.t fc layers and generator
        trainable_weights = self.FC.trainable_weights + self.G.trainable_weights
        g_grad = tape.gradient(g_loss, trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grad, trainable_weights))
        
        return {'d_loss': d_lost, 'g_loss': g_loss}
