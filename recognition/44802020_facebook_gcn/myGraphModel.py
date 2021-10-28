import keras.initializers.initializers_v1
import scipy
import scipy.sparse as spr
import tensorflow as tf
import tensorflow.keras.layers as lyr
import tensorflow.keras as ks


@tf.function
def printIncident(incident):
    print("MyFunc")
    print(incident)
    return incident


class FaceGCNLayer(tf.keras.layers.Layer):
    def __init__(self, adj_m):
        super(FaceGCNLayer, self).__init__()
        self.adj_m = adj_m

    def build(self, input_shape):
        self.weights1 = self.add_weight("weights1",
                                       shape=input_shape[1:],
                                       initializer=keras.initializers.initializers_v1.RandomNormal)

    def call(self, feature_matrix):
        feature_matrix = tf.squeeze(feature_matrix)
        ax = tf.sparse.sparse_dense_matmul(tf.cast(self.adj_m, float), feature_matrix)
        z = ax * self.weights1

        return z

