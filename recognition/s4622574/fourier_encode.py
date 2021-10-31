from tensorflow.keras import layers
import tensorflow as tf 
import math

class FourierEncode(layers.Layer):
    def __init__(self, max_freq=10, freq_ban=4):
        super(FourierEncode, self).__init__()
        self.max_freq = max_freq
        self.freq_ban = freq_ban

    def call(self, patientData):
        
        num_images, *basis, _ = patientData.shape

        basis = tuple(basis)

        basis_feature = list(map(lambda x: tf.linspace(-1.0, 1.0, num=x), basis))

        fearture = tf.stack(tf.meshgrid(*basis_feature, indexing="ij"), axis=-1)


        transformedFeature = self._fourier_encode(fearture)

        del fearture
        
        transformedFeature = tf.reshape(transformedFeature, (1, basis[0], basis[1], 2*(2*self.freq_ban+1)))


        transformedFeature = tf.repeat(transformedFeature, repeats=num_images, axis=0)

        transformedData = tf.concat((patientData, transformedFeature), axis=-1)


        transformedData = tf.reshape(transformedData, (num_images, basis[0]*basis[1], -1)) 
        return transformedData




    def _fourier_encode(self, fearture):

        fearture = tf.expand_dims(fearture, -1)
        fearture = tf.cast(fearture, dtype=tf.float32)
        sampleFeature = fearture
        
        scaledKernel = tf.experimental.numpy.logspace(start=0.0, 
                stop=math.log(self.max_freq / 2) / math.log(10),
                num=self.freq_ban, dtype=tf.float32,
        )

        scaledKernel = tf.reshape(scaledKernel, (*((1,) * (len(fearture.shape) - 1)), self.freq_ban))


        fearture = fearture * scaledKernel * math.pi

        fearture = tf.concat([tf.math.sin(fearture), tf.math.cos(fearture)], axis=-1)
        fearture = tf.concat((fearture, sampleFeature), axis=-1)
        return fearture