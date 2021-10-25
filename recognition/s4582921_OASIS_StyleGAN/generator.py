import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from layers import Conv2DModulation

LATENT_SIZE = 512
KERNEL_SIZE = 3
ALPHA = 0.2
BETA = 0.999

class Generator():


    def __init__(self, image_size, blocks, learning_rate, channels):

        self.kernel_size = (KERNEL_SIZE, KERNEL_SIZE)

        self.image_size = image_size
        self.blocks = blocks
        self.channels = channels

        self.style = self.build_style()
        self.model = self.build_model()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate, 0, BETA)


    def train_block(self, input, style, noise, filters):

        output = input

        style_output = Dense(filters)(style)
        style1 = Dense(input.shape[-1])(style)
        style2 = Dense(filters)(style)

        delta = Lambda(lambda x: x[0][:, :x[1].shape[1], :x[1].shape[2], :])([noise, output])
        delta1 = Dense(filters)(delta)
        delta2 = Dense(filters)(delta)

        output = Conv2DModulation(filters=filters, kernel_size=self.kernel_size, padding='same')([output, style1])
        output = add([output, delta1])
        output = LeakyReLU(ALPHA)(output)

        output = Conv2DModulation(filters=filters, kernel_size=self.kernel_size, padding='same')([output, style2])
        output = add([output, delta2])
        output = LeakyReLU(ALPHA)(output)

        image = self.get_image(output, style_output)

        return output, image


    def build_model(self):

        outputs = []
        
        styles = [Input([LATENT_SIZE]) for block in range(self.blocks)]
        noise = Input([self.image_size, self.image_size, 1])
        input = Lambda(lambda x: x[:, :1] * 0 + 1)(styles[0])

        output = Dense(4 * 4 * 2**(self.blocks) * self.channels, activation='relu')(input)
        output = Reshape([4, 4, 2**(self.blocks) * self.channels])(output)
        output = Activation('linear')(output)

        for block in reversed(range(self.blocks)):
            output, image = self.train_block(output, styles[block], noise, self.channels * 2**block)
            if block != 0:
                output = UpSampling2D(interpolation='bilinear')(output)
            outputs.append(image)

        output = add(outputs)

        return Model(inputs=styles + [noise], outputs=output)


    def build_style(self):

        input = Input([LATENT_SIZE])
        output = input

        for block in range(4):
            output = Dense(LATENT_SIZE)(output)
            output = LeakyReLU(ALPHA)(output)

        return Model(inputs=input, outputs=output)


    def loss(self, fake_output):

        return tf.keras.losses.BinaryCrossentropy(tf.ones_like(fake_output), fake_output)

        
    def w_loss(self, fake_output):

        return -tf.reduce_mean(fake_output)


    def get_image(self, input, style):

        size = input.shape[2]
        image = Conv2DModulation(filters=1, kernel_size=1, padding='same', demod=False)([input, style])

        scale = int(self.image_size / size)

        def rescale(x, y=scale):
            return tf.keras.backend.resize_images(x, y, y, "channels_last",interpolation='bilinear')
            
        image = Lambda(rescale, output_shape=[None, self.image_size, self.image_size, None])(image)

        return image
        





