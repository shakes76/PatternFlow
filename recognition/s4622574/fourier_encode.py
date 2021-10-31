from tensorflow.keras import layers
import tensorflow as tf 
import math

class FourierEncode(layers.Layer):
    def __init__(self, max_freq=10, freq_ban=4):
        super(FourierEncode, self).__init__()
        self.max_freq = max_freq
        self.freq_ban = freq_ban

    def encoding(transformedFeature, basis, fred_ban, num_images):
        transformedFeature = tf.reshape(transformedFeature, (1, basis[0], basis[1], 2 * (2 * freq_ban + 1)))

        return tf.repeat(transformedFeature, repeats=num_images, axis=0)


    def call(self, patientData):
        
        # num_images, *basis, _ = patientData.shape

        num_images, *basis, tempt = patientData.shape
        

        basis = tuple(basis)

        basis_feature = list(map(lambda x: tf.linspace(-1.0, 1.0, num=x), basis))

        feature = tf.stack(tf.meshgrid(*basis_feature, indexing="ij"), axis=-1)


        transformedFeature = self._fourier_encode(feature)

        # del feature

        # transformedFeature = tf.reshape(transformedFeature, (1, basis[0], basis[1], 2 * (2 * self.freq_ban + 1)))


        # transformedFeature = tf.repeat(transformedFeature, repeats=num_images, axis=0)

        transformedFeature = self.encoding(transformedFeature, basis, self.freq_ban, num_images)

        # transformedData = tf.concat((patientData, transformedFeature), axis=-1)


        # transformedData = tf.reshape(transformedData, (num_images, basis[0]*basis[1], -1)) 
        transformedData = self.getCombinedData(patientData, transformedFeature, num_images, basis)
        return transformedData

    def getCombinedData(patientData, transformedFeature, basis, num_images):
        transformedData = tf.concat((patientData, transformedFeature), axis=-1)


        return tf.reshape(transformedData, (num_images, basis[0] * basis[1], -1)) 
        




    def _fourier_encode(self, feature):

        # feature = tf.expand_dims(feature, -1)
        feature = tf.cast(tf.expand_dims(feature, -1), dtype=tf.float32)
        sampleFeature = feature
        
        scaledKernel = tf.experimental.numpy.logspace(start=0.0, 
                stop=math.log(self.max_freq / 2) / math.log(10),
                num=self.freq_ban, dtype=tf.float32,
        )

        scaledKernel = tf.reshape(scaledKernel, (*((1,) * (len(feature.shape) - 1)), self.freq_ban))


        feature = math.pi * scaledKernel * feature 

        feature = tf.concat([tf.math.sin(feature), tf.math.cos(feature)], axis=-1)
        feature = tf.concat((feature, sampleFeature), axis=-1)
        return feature

