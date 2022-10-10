import numpy as np
import tensorflow as tf
from keras import backend, layers
from keras.initializers import RandomNormal
from keras.layers import Add, Dense, Flatten, Input, Layer, LeakyReLU
from keras.models import Model

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
        stddev = tf.sqrt(tf.reduce_mean(tf.square(inputs - mean), axis=0, keepdims=True) + 1e-8)
        average_stddev = tf.reduce_mean(stddev, keepdims=True)
        shape = tf.shape(inputs)
        minibatch_stddev = tf.tile(average_stddev, (shape[0], shape[1], shape[2], 1))
        combined = tf.concat([inputs, minibatch_stddev], axis=-1)
        return combined

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)

# Weighted Sum
# Perform Weighted Sum
# Define alpha as backend.variable to update during training
class WeightedSum(Add):
    
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')

    def _merge_function(self, inputs):
        assert(len(inputs) == 2)
        output = ((1.0 - self.alpha) * inputs[0] + (self.alpha * inputs[1]))
        return output

# normalize weight by given shape
# https://prateekvishnu.medium.com/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
# stddev = sqrt(2 / fan_in)
class WeightScaling(Layer):
    
    def __init__(self, shape, gain=np.sqrt(2), **kwargs):
        super(WeightScaling, self).__init__(**kwargs)
        shape = np.asarray(shape)
        shape = tf.constant(shape, dtype=tf.float32)
        fan_in = tf.math.reduce_prod(shape)
        self.wscale = gain/tf.math.sqrt(fan_in)

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.float32)
        return inputs * self.wscale

    def compute_output_shape(self, input_shape):
        return input_shape


class Bias(Layer):
    
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(shape=(input_shape[-1],), dtype='float32'), trainable=True)

    def call(self, inputs, **kwargs):
        return inputs + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


def EqualDense(x, filters, gain, activate=None):
    init = RandomNormal(mean=0., stddev=1.)
    in_filters = backend.int_shape(x)[-1]
    x = layers.Dense(filters, use_bias=False, kernel_initializer=init, dtype='float32')(x)
    x = WeightScaling(shape=(in_filters), gain=gain)(x)
    x = Bias(input_shape=x.shape)(x)
    if activate == 'LeakyReLU':
        x = layers.LeakyReLU(0.2)(x)
    elif activate == 'tanh':
        x = layers.Activation('tanh')(x)
    return x


def EqualConv(x, filters, gain, activate=None, kernel_size=(3, 3), strides=(1, 1)):
    init = RandomNormal(mean=0., stddev=1.)
    in_filters = backend.int_shape(x)[-1]
    x = layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", kernel_initializer=init, dtype='float32')(x)
    x = WeightScaling(shape=(kernel_size[0], kernel_size[1], in_filters), gain=gain)(x)
    x = Bias(input_shape=x.shape)(x)
    if activate == 'LeakyReLU':
        x = LeakyReLU(0.2)(x)
    elif activate == 'tanh':
        x = layers.Activation('tanh')(x)
    return x

# adaptive instance normalization
class AdaIN(layers.Layer):

    def __init__(self, **kwargs):
        super(AdaIN, self).__init__(**kwargs)

    def build(self, input_shapes):
        filters = input_shapes[0][-1]
        self.denseys = Dense(filters)
        self.denseyb = Dense(filters)
        self.wsys = WeightScaling(shape=(filters))
        self.wsyb = WeightScaling(shape=(filters))

    def call(self, inputs):
        x, w = inputs
        
        # instance norm
        m, v = tf.nn.moments(x, [1, 2], keepdims=True)
        x = (x-m)/tf.sqrt(v+1.0e-8)

        # w -> A
        ys = self.wsys(self.denseys(w))
        yb = self.wsyb(self.denseyb(w))
        
        # reshape for calculation
        ys = tf.reshape(ys, (-1, 1, 1, x.shape[-1]))
        yb = tf.reshape(yb, (-1, 1, 1, x.shape[-1]))

        return ys * x + yb


class AddNoise(layers.Layer):
    
    def build(self, input_shape):
        filters = input_shape[0][-1]
        init = RandomNormal(mean=0., stddev=1.)
        # without specifying 'name' excption raised when saving weights. bug?
        self.b = self.add_weight(name='w', shape=[1, 1, 1, filters], initializer=init, trainable=True)
        self.lrelu = LeakyReLU(0.2)
    def call(self, inputs):
        x, B = inputs
        out = x + self.b * B
        out = self.lrelu(out)
        return out


class StyleGAN(Model):

    def __init__(
        self,
        latent_dim,
        filters,
        gp_weight=10.0,
        drift_weight=0.001,
        sres=4,
        tres=256,
        channels=1
    ):
        super(StyleGAN, self).__init__()
        
        self.LDIM = latent_dim
        self.CHANNELS = channels                  # image channels
        self.FILTERS = filters                    # set of filters
        depth = int(np.log2(tres)-np.log2(sres))
        self.DEPTH = depth                        # training depth
        self.SRES = sres                          # start resolution
        self.TRES = tres                          # target resolution
        
        self.gp_weight = gp_weight
        self.drift_weight = drift_weight
        
        self.current_depth = 0                    # current training depth
        
        self.FC = self.mapping()                  # FC net
        self.G = self.init_G()                    # generator
        self.D = self.init_D()                    # discriminator
    
    def compile(self, d_optimizer, g_optimizer):
        super(StyleGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        
    # fully connected layers. z->w.
    def mapping(self):
        z = Input(shape=(self.LDIM))
        # 8 layers in paper. use 6 instead.
        w = EqualDense(z, filters=self.LDIM, gain=1., activate='LeakyReLU')
        for _ in range(self.DEPTH-1):
            w = EqualDense(w, filters=self.LDIM, gain=1., activate='LeakyReLU')
        w = tf.tile(tf.expand_dims(w, 1), (1, self.DEPTH+1, 1)) # (256,7)
        return Model(z, w)

    def init_D(self):
        image = Input(shape=(self.SRES, self.SRES, self.CHANNELS))
        x = EqualConv(image, filters=self.FILTERS[0], kernel_size=(1, 1), gain=np.sqrt(2), activate='LeakyReLU')
        x = MinibatchStdev()(x)
        x = EqualConv(x, filters=self.FILTERS[0], kernel_size=(3, 3), gain=np.sqrt(2), activate='LeakyReLU')
        x = EqualConv(x, filters=self.FILTERS[0], kernel_size=(4, 4), gain=np.sqrt(2), activate='LeakyReLU', strides=(4, 4))
        x = EqualDense(Flatten()(x), filters=self.CHANNELS, gain=1.)
        d_model = Model(image, x)
        return d_model

    # grow discriminator
    def grow_D(self):
        input_shape = list(self.D.input.shape)
        
        # w*2, h*2, c 
        input_shape = (input_shape[1]*2, input_shape[2]*2, input_shape[3])
        image = Input(shape=input_shape)
        image = tf.cast(image, tf.float32)

        # downsample
        x1 = layers.AveragePooling2D()(image)
        for i in [1, 2, 3, 4]:
            x1 = self.D.layers[i](x1)

        # fade in
        x2 = EqualConv(image, filters=self.FILTERS[self.current_depth], kernel_size=(1, 1), gain=np.sqrt(2), activate='LeakyReLU')
        x2 = EqualConv(x2, filters=self.FILTERS[self.current_depth], kernel_size=(3, 3), gain=np.sqrt(2), activate='LeakyReLU')
        x2 = EqualConv(x2, filters=self.FILTERS[self.current_depth-1], kernel_size=(3, 3), gain=np.sqrt(2), activate='LeakyReLU')
        x2 = layers.AveragePooling2D()(x2)
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
        r = self.SRES
        f = self.FILTERS[0]
        const = Input(shape=(r, r, f), name='const')
        w = Input(shape=(self.LDIM), name='w')
        B = Input(shape=(r, r, 1), name='B')
        x = const
        
        x = AddNoise()([x, B])
        x = AdaIN()([x, w])
        x = EqualConv(x, filters=f, gain=np.sqrt(2), activate='LeakyReLU')
        
        x = AddNoise()([x, B])
        x = AdaIN()([x, w])
        x = EqualConv(x, filters=self.CHANNELS, kernel_size=(1, 1), gain=1., activate='tanh')
        return Model([const, w, B], x)

    # Fade in upper resolution block
    def grow_G(self):
        sqrt2 = np.sqrt(2)
        res = self.SRES*(2**self.current_depth) 
        end = self.G.layers[-5].output
        end = layers.UpSampling2D((2, 2))(end)

        # branch
        x1 = end
        for i in [-4, -3, -2, -1]:
            x1 = self.G.layers[i](x1)

        # branch
        w = Input(shape=(self.LDIM))
        B = Input(shape=(res, res, 1))
        
        x2 = EqualConv(end, filters=self.FILTERS[self.current_depth], gain=sqrt2, activate='LeakyReLU')
        x2 = AddNoise()([x2, B])
        x2 = AdaIN()([x2, w])
        
        x2 = EqualConv(end, filters=self.FILTERS[self.current_depth], gain=sqrt2, activate='LeakyReLU')
        x2 = AddNoise()([x2, B])
        x2 = AdaIN()([x2, w])
        
        # to rgb
        x2 = EqualConv(x2, filters=self.CHANNELS, kernel_size=(1, 1), gain=1., activate='tanh')

        # stabilize
        self.G_ST = Model(self.G.input+[w,B], x2)
        
        # fade in
        self.G = Model(self.G.input+[w,B], WeightedSum()([x1, x2]))

    def grow(self):
        self.current_depth += 1
        self.grow_G()
        self.grow_D()

    def stabilize(self):
        self.G = self.G_ST
        self.D = self.D_ST

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """
        gradient penalty on an interpolated image, added to the discriminator loss
        """
        # Get the interpolated image
        alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            # discriminator output of this interpolated image
            pred = self.D(interpolated, training=True)

        # gradients w.r.t to interpolated image
        grads = tape.gradient(pred, [interpolated])[0]
        # norm of the gradient
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_images = data[0]
        batch_size = tf.shape(real_images)[0]
        const = tf.ones([batch_size, self.SRES, self.SRES, self.FILTERS[0]])

        # train discriminator
        with tf.GradientTape() as tape:
            
            # build input for G: [const, w, B, w, B, w , B, ...]
            z = tf.random.normal(shape=(batch_size, self.LDIM))
            ws = self.FC(z)
            inputs = [const]
            for i in range(self.current_depth+1):
                w = ws[:,i]
                B = tf.random.normal((batch_size, self.SRES*(2**i), self.SRES*(2**i), 1))
                inputs += [w, B]
            
            fake_images = self.G(inputs, training=True)
            fake_labels = self.D(fake_images, training=True)
            real_labels = self.D(real_images, training=True)

            d_cost = tf.reduce_mean(fake_labels) - tf.reduce_mean(real_labels)
            gp = self.gradient_penalty(batch_size, real_images, fake_images)
            
            # drift for regularization
            drift = tf.reduce_mean(tf.square(real_labels))

            # gradient penalty to dloss
            d_loss = d_cost + self.gp_weight * gp + self.drift_weight * drift

        # gradients w.r.t dloss
        d_grad = tape.gradient(d_loss, self.D.trainable_weights)
        # update discriminator weights
        self.d_optimizer.apply_gradients(zip(d_grad, self.D.trainable_weights))

        # train generator
        with tf.GradientTape() as tape:
            z = tf.random.normal(shape=(batch_size, self.LDIM))
            ws = self.FC(z)
            inputs = [const]
            for i in range(self.current_depth+1):
                w = ws[:,i]
                B = tf.random.normal((batch_size, self.SRES*(2**i), self.SRES*(2**i), 1))
                inputs += [w, B]
            generated_images = self.G(inputs, training=True)
            pred_labels = self.D(generated_images, training=True)
            # wasserstein distance
            g_loss = -tf.reduce_mean(pred_labels)
        # gradients w.r.t fully connected layers and generator
        trainable_weights = (self.FC.trainable_weights + self.G.trainable_weights)
        g_grad = tape.gradient(g_loss, trainable_weights)
        # update the weights
        self.g_optimizer.apply_gradients(zip(g_grad, trainable_weights))
        return {'d_loss': d_loss, 'g_loss': g_loss}
