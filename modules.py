import numpy as np
import tensorflow as tf
from keras import backend, layers
from keras.initializers import RandomNormal
from keras.layers import Add, Layer, Dense, Flatten
from keras.models import Model

# Mini Batch Standadization
# Calculate the average standard deviation of all features and spatial location.
# Concat after creating a constant feature map with the average standard deviation


class MinibatchStdev(Layer):
    
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    def call(self, inputs):
        # Mean accross channels
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        # Std accross channels
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
        self.wscale = gain*tf.math.rsqrt(fan_in)

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
        self.bias = tf.Variable(initial_value=b_init(
            shape=(input_shape[-1],), dtype='float32'), trainable=True)

    def call(self, inputs, **kwargs):
        return inputs + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


def WeightScalingDense(x, filters, gain, activate=None):
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


def WeightScalingConv(x, filters, gain, activate=None, kernel_size=(3, 3), strides=(1, 1)):
    init = RandomNormal(mean=0., stddev=1.)
    in_filters = backend.int_shape(x)[-1]
    x = layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", kernel_initializer=init, dtype='float32')(x)
    x = WeightScaling(shape=(kernel_size[0], kernel_size[1], in_filters), gain=gain)(x)
    x = Bias(input_shape=x.shape)(x)
    if activate == 'LeakyReLU':
        x = layers.LeakyReLU(0.2)(x)
    elif activate == 'tanh':
        x = layers.Activation('tanh')(x)
    return x


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
        x = (x - m) / tf.sqrt(v + 1.0e-8)

        # w -> A
        ys = self.wsys(self.denseys(w))
        yb = self.wsyb(self.denseyb(w))
        
        # reshape for calculation
        ys = tf.reshape(ys, (-1, 1, 1, x.shape[-1]))
        yb = tf.reshape(yb, (-1, 1, 1, x.shape[-1]))

        return ys * x + yb


class AddNoise(layers.Layer):
    
    def build(self, input_shape):
        channels = input_shape[0][-1]
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.b = self.add_weight(shape=[1, 1, 1, channels], initializer=initializer, trainable=True, name="kernel")

    def call(self, inputs):
        x, noise = inputs
        output = x + self.b * noise
        return output


class StyleGAN(Model):

    def __init__(
        self,
        latent_dim,
        filters,
        gp_weight=10.0,
        drift_weight=0.001,
        sres=4,
        eres=256,
        channels=1
    ):
        super(StyleGAN, self).__init__()
        
        self.LDIM = latent_dim
        self.CHANNELS = channels                  # image channels
        self.FILTERS = filters                    # set of filters
        depth = int(np.log2(eres)-np.log2(sres))
        self.DEPTH = depth                        # training depth
        self.SRES = sres                          # start resolution
        self.TRES = eres                          # target resolution
        
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
        
    def mapping(self):
        z = layers.Input(shape=(self.LDIM))
        # 8 layers in paper
        for _ in range(self.DEPTH):
            w = WeightScalingDense(z, filters=self.LDIM, gain=1., activate='LeakyReLU')
        w = tf.tile(tf.expand_dims(w, 1), (1, self.DEPTH+1, 1))
        return Model(z, w)

    def init_D(self):
        rgb = layers.Input(shape=(self.SRES, self.SRES, self.CHANNELS))
        x = WeightScalingConv(rgb, filters=self.FILTERS[0], kernel_size=(1, 1), gain=np.sqrt(2), activate='LeakyReLU')
        x = MinibatchStdev()(x)
        x = WeightScalingConv(x, filters=self.FILTERS[0], kernel_size=(3, 3), gain=np.sqrt(2), activate='LeakyReLU')
        x = WeightScalingConv(x, filters=self.FILTERS[0], kernel_size=(4, 4), gain=np.sqrt(2), activate='LeakyReLU', strides=(4, 4))
        x = WeightScalingDense(Flatten()(x), filters=self.CHANNELS, gain=1.)
        d_model = Model(rgb, x, name='discriminator')
        return d_model

    # Fade in upper resolution block
    def grow_D(self):
        # for layer in self.discriminator.layers:
        #    layer.trainable = False
        input_shape = list(self.D.input.shape)
        # 1. Double the input resolution.
        input_shape = (input_shape[1]*2, input_shape[2]*2, input_shape[3])
        img_input = layers.Input(shape=input_shape)
        img_input = tf.cast(img_input, tf.float32)

        # 2. Add pooling layer
        #    Reuse the existing “formRGB” block defined as “x1".
        x1 = layers.AveragePooling2D()(img_input)
        for i in [1, 2, 3, 4]:
            x1 = self.D.layers[i](x1)

        # 3.  Define a "fade in" block (x2) with a new "fromRGB" and two 3x3 convolutions.
        #     Add an AveragePooling2D layer
        x2 = WeightScalingConv(img_input, filters=self.FILTERS[self.current_depth], kernel_size=(1, 1), gain=np.sqrt(2), activate='LeakyReLU')
        x2 = WeightScalingConv(x2, filters=self.FILTERS[self.current_depth], kernel_size=(3, 3), gain=np.sqrt(2), activate='LeakyReLU')
        x2 = WeightScalingConv(x2, filters=self.FILTERS[self.current_depth-1], kernel_size=(3, 3), gain=np.sqrt(2), activate='LeakyReLU')

        x2 = layers.AveragePooling2D()(x2)
        x = WeightedSum()([x1, x2])

        # Define stabilized(c. state) discriminator
        for i in range(5, len(self.D.layers)):
            x2 = self.D.layers[i](x2)
        self.DT = Model(img_input, x2, name='discriminator')

        # 5. Add existing discriminator layers.
        for i in range(5, len(self.D.layers)):
            x = self.D.layers[i](x)
        self.D = Model(img_input, x, name='discriminator')

    def init_G(self):
        const = layers.Input(shape=(self.SRES, self.SRES, self.FILTERS[0]))
        w = layers.Input(shape=(self.DEPTH+1, self.LDIM))
        x = const
        x = AdaIN()([x, w[:, 0]])
        x = WeightScalingConv(x, filters=self.FILTERS[0], gain=np.sqrt(2), activate='LeakyReLU')
        x = AdaIN()([x, w[:, 0]])
        x = WeightScalingConv(x, filters=self.CHANNELS, kernel_size=(1, 1), gain=1., activate='tanh')
        return Model([const, w], x)

    # Fade in upper resolution block
    def grow_G(self):
        sqrt2 = np.sqrt(2)
        
        end = self.G.layers[-5].output
        end = layers.UpSampling2D((2, 2))(end)

        # branch
        # x1 = end
        # for i in [-4, -3, -2, -1]:
        #     x1 = self.G.layers[i](x1)
                
        x1 = self.G.layers[-4](end) # Conv2d
        x1 = self.G.layers[-3](x1)  # WeightScalingLayer
        x1 = self.G.layers[-2](x1)  # Bias
        x1 = self.G.layers[-1](x1)  # tanh

        # branch
        w = self.G.input[1]
        x2 = WeightScalingConv(end, filters=self.FILTERS[self.current_depth], gain=sqrt2, activate='LeakyReLU')
        x2 = AdaIN()([x2, w[:, self.current_depth]])
        x2 = WeightScalingConv(end, filters=self.FILTERS[self.current_depth], gain=sqrt2, activate='LeakyReLU')
        x2 = AdaIN()([x2, w[:, self.current_depth]])
        x2 = WeightScalingConv(x2, filters=self.CHANNELS, kernel_size=(1, 1), gain=1., activate='tanh')

        self.GT = Model(self.G.input, x2)
        x = WeightedSum()([x1, x2])
        self.G = Model(self.G.input, x)

    def grow(self):
        self.grow_G()
        self.grow_D()

    def transition(self):
        self.G = self.GT
        self.D = self.DT

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.D(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_images = data[0]
        batch_size = tf.shape(real_images)[0]
        const = tf.ones([batch_size, self.SRES, self.SRES, self.FILTERS[0]])
        
        # Get the latent vector
        z = tf.random.normal(shape=(batch_size, self.LDIM))
        with tf.GradientTape() as tape:
            w = self.FC(z)
            # Generate fake images from the latent vector
            fake_images = self.G([const, w], training=True)
            # Get the logits for the fake images
            fake_labels = self.D(fake_images, training=True)
            # Get the logits for the real images
            real_labels = self.D(real_images, training=True)

            # Calculate the discriminator loss using the fake and real image logits
            d_cost = tf.reduce_mean(fake_labels) - tf.reduce_mean(real_labels)

            # Calculate the gradient penalty
            gp = self.gradient_penalty(batch_size, real_images, fake_images)
            # Calculate the drift for regularization
            drift = tf.reduce_mean(tf.square(real_labels))

            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + self.gp_weight * gp + self.drift_weight * drift

        # Get the gradients w.r.t the discriminator loss
        d_grad = tape.gradient(d_loss, self.D.trainable_weights)
        # Update the weights of the discriminator using the discriminator optimizer
        self.d_optimizer.apply_gradients(zip(d_grad, self.D.trainable_weights))

        # Train the generator
        # Get the latent vector
        z = tf.random.normal(shape=(batch_size, self.LDIM))
        with tf.GradientTape() as tape:
            w = self.FC(z)
            # Generate fake images using the generator
            generated_images = self.G([const, w], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.D(generated_images, training=True)
            # Calculate the generator loss
            g_loss = -tf.reduce_mean(gen_img_logits)
        # Get the gradients w.r.t the generator loss
        trainable_weights = (self.G.trainable_weights + self.FC.trainable_weights)
        g_grad = tape.gradient(g_loss, trainable_weights)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(g_grad, trainable_weights))
        return {'d_loss': d_loss, 'g_loss': g_loss}
