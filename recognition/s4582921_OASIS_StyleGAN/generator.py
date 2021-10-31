import tensorflow as tf
from tensorflow.keras.backend import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import gan
from layers import Conv2DModulation


#Hyper Parameters
KERNEL_SIZE = 3
ALPHA = 0.2
BETA = 0.999


class Generator():
    """
    The generator model which attempts to fool the discriminator by producing fake images.

    Attributes:
        image_size : the size of images being classified
        blocks : number of train blocks used in building the model
        learning_rate : learning rate of the optimizer
        channels : channels in image used for training
        kernel_size : size of the kernel in convolutional layers
        style : reference to the learned style
        model : reference to the generator model
        optimizer : optimizer used in the generator model
    """


    def __init__(self, image_size, blocks, learning_rate, channels):
        """Initialises an instance of the Generator"""

        self.kernel_size = (KERNEL_SIZE, KERNEL_SIZE)

        self.image_size = image_size
        self.blocks = blocks
        self.channels = channels

        self.style = self.build_style()
        self.model = self.build_model()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate, 0, BETA)


    def train_block(self, input, style, noise, filters):
        """
        A training block used in building the model.

        Args:
            input : previous layer as input
            style : latent style input used to generate images
            noise : intermediate noise to insert
            filters : number of filters to use in convolution layers
        Returns:
            The output layer produced by the block.
        """

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
        """
        Builds the generator model.

        Returns:
            The model for the generator.
        """

        outputs = []
        
        styles = [Input([gan.LATENT_SIZE]) for block in range(self.blocks)]
        noise = Input([self.image_size, self.image_size, 1])
        input = Lambda(lambda x: x * 0 + 1)(styles[0])

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
        """
        Builds the style model.

        Returns:
            The style model for the generator.
        """

        input = Input([gan.LATENT_SIZE])
        output = input

        for block in range(4):
            output = Dense(gan.LATENT_SIZE)(output)
            output = LeakyReLU(ALPHA)(output)

        return Model(inputs=input, outputs=output)


    def loss(self, fake_output):
        """
        Basic binary cross entropy loss.

        Args:
            fake_output : classification of a batch of fake images
        Returns:
            The loss.
        """

        criterion = tf.keras.losses.BinaryCrossentropy()
        return criterion(tf.ones_like(fake_output), fake_output)


    def l_loss(self, fake_output):
        """
        Logistic loss used in official implementation.

        Args:
            fake_output : classification of a batch of fake images
        Returns:
            The loss.
        """

        return -tf.nn.softplus(fake_output)
        

    def w_loss(self, fake_output):
        """
        Wasserstein loss used in official implementation.

        Args:
            fake_output : classification of a batch of fake images
        Returns:
            The loss.
        """

        return -tf.reduce_mean(fake_output)


    def h_loss(self, fake_output):
        """
        Hinge loss.

        Args:
            fake_output : classification of a batch of fake images
        Returns:
            The loss.
        """

        return mean(fake_output)


    def get_image(self, input, style):
        """
        Get the image at current time to preserve broader features despite image upscaling.

        Args:
            input : the current layer used as input
            style : the persistant style to use
        Returns:
            Image representaion of the current state of the model.
        """

        size = input.shape[2]
        image = Conv2DModulation(filters=1, kernel_size=1, padding='same', demod=False)([input, style])

        scale = int(self.image_size / size)

        def rescale(x, y=scale):
            return resize_images(x, y, y, "channels_last",interpolation='bilinear')
            
        image = Lambda(rescale, output_shape=[None, self.image_size, self.image_size, None])(image)

        return image
        





