
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, InputLayer, Flatten, Dense, Reshape, BatchNormalization, Dropout


class VQVAE(tf.keras.Model):

    def __init__(self, latent_dimension, kernel_size=3, strides=2):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.vector_quant = self.build_vector_quantizer()

    def build_encoder(self, latent_dimension, kernel_size, strides):
        """Encoder is based off my Demo 2 VAE submission with some minor changes"""

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(256,256,1)))

        model.add(tf.keras.layers.Conv2D(64, kernel_size, strides, 'same', activation='relu')) # 'same' padding makes sure entire image is convoluted
        model.add(tf.keras.layers.BatchNormalization()) #recentering 
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2D(64, kernel_size, strides, 'same', activation='relu')) # 'same' padding makes sure entire image is convoluted
        model.add(tf.keras.layers.BatchNormalization()) #recentering 
        model.add(tf.keras.layers.LeakyReLU())

        # compress image
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(latent_dimension * 2)) 

        print(model.summary)
        return model

    def build_decoder(self, latent_dimension, kernel_size, strides):
        """Decoder is based off my Demo 2 VAE submission with some minor changes"""

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(latent_dimension,)))
        model.add(tf.keras.layers.Dense(8*8*64))
        model.add(tf.keras.layers.Reshape((8,8,64)))

        model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size, strides, 'same', activation='relu'))
        model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size, strides, 'same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization()) #recentering 
        model.add(tf.keras.layers.LeakyReLU()) # to avoid dead weights 

        model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size, strides, 'same', activation='relu'))
        model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size, strides, 'same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization()) #recentering
        model.add(tf.keras.layers.LeakyReLU()) # to avoid dead weights 

        model.add(tf.keras.layers.Conv2DTranspose(3, kernel_size, strides, 'same', activaiton='sigmoid'))

        print(model.summary)
        return model

    def build_vector_quantizer(self, z):
        """
        z: continuous input (output of encoder)

        Returns z_quantized (discrete variable)

        shape of z: (batch_size, img_height, img_width, 1)

        We want to map z to a discrete 1 hot encoded vector which indexse
        """

        #discrete instead of continuous
        #1 hot encoding which indexes closest e
        pass


        # flatten

        # calc euclidean distance between 

        # find closest encoding

    def compute_loss(self):
        pass


    def forward(self, x):
        """Method modified versin from https://github.com/MishaLaskin/vqvae/blob/master/models/vqvae.py"""
        z_e = self.encoder(x)
        embedding_loss, z_q, perplex = self.vector_quant()

