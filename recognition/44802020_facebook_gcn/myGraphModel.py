import keras.initializers.initializers_v1
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout


def makeMyModel(a_bar, a_bar_test, train_feats):
    my_model = Sequential()

    my_model.add(Input(shape=tf.Tensor.get_shape(train_feats)))

    my_model.add(FaceGCNLayer(a_bar, a_bar_test))
    my_model.add(Dropout(0.4))
    my_model.add(Dense(64))
    my_model.add(FaceGCNLayer(a_bar, a_bar_test))
    my_model.add(Dropout(0.4))
    my_model.add(Dense(32))
    my_model.add(FaceGCNLayer(a_bar, a_bar_test))
    my_model.add(Dropout(0.4))
    my_model.add(Dense(4, activation='softmax'))

    return my_model


class FaceGCNLayer(tf.keras.layers.Layer):
    def __init__(self, adj_m, test_adj_m):
        super(FaceGCNLayer, self).__init__()
        self.adj_m = adj_m
        self.test_adj_m = test_adj_m

    def build(self, input_shape):
        self.weights1 = self.add_weight("weights1",
                                       shape=(1, input_shape[-1]),
                                       initializer=keras.initializers.initializers_v1.RandomNormal)

    def call(self, feature_matrix, training=None):
        feature_matrix = tf.squeeze(feature_matrix)
        if training:
            ax = tf.sparse.sparse_dense_matmul(tf.cast(self.adj_m, float), feature_matrix)
            z = ax * self.weights1
        else:
            ax = tf.sparse.sparse_dense_matmul(tf.cast(self.test_adj_m, float), feature_matrix)
            z = ax * self.weights1

        return z
