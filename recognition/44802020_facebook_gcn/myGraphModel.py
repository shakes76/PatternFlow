import keras.initializers.initializers_v1
import scipy
import scipy.sparse as spr
import tensorflow as tf


@tf.function
def printIncident(incident):
    print("MyFunc")
    print(incident)
    return incident


class FaceGCNLayer(tf.keras.layers.Layer):
    def __init__(self, adj_m, test_adj_m):
        super(FaceGCNLayer, self).__init__()
        self.adj_m = adj_m
        self.test_adj_m = test_adj_m
        # self.weights1 = []
        # self.build(input_shape=input_shape)

    def build(self, input_shape):
        print("Building")
        print(input_shape)
        print(input_shape[-1])
        #1:
        self.weights1 = self.add_weight("weights1",
                                       shape=(1, input_shape[-1]),
                                       initializer=keras.initializers.initializers_v1.RandomNormal)

    def call(self, feature_matrix, training=None):
        print(training)
        feature_matrix = tf.squeeze(feature_matrix)
        if training:
            ax = tf.sparse.sparse_dense_matmul(tf.cast(self.adj_m, float), feature_matrix)
            z = ax * self.weights1
        else:
            ax = tf.sparse.sparse_dense_matmul(tf.cast(self.test_adj_m, float), feature_matrix)
            z = ax * self.weights1

        return z


class MyModel(tf.keras.Model):

    def __init__(self, adj, eval_adj, input_shape):
        super(MyModel, self).__init__()
        self.adj = adj
        self.eval_adj = eval_adj
        self.my_input_shape = input_shape

        self.input1 = tf.keras.layers.Input(shape=self.my_input_shape)

        self.dense1 = tf.keras.layers.Dense(96)
        self.gcn1 = FaceGCNLayer(self.adj, input_shape)
        self.dense2 = tf.keras.layers.Dense(64)
        self.gcn2 = FaceGCNLayer(self.adj, input_shape)
        self.dense3 = tf.keras.layers.Dense(4)

    def set_input_shape(self, new_shape):
        self.my_input_shape = new_shape
        self.input1 = tf.keras.layers.Input(shape=self.my_input_shape)

    def call(self, inputs):
        # x = tf.keras.layers.Input(inputs)
        x = self.dense1(inputs)
        x = self.gcn1.call(x, self.adj)
        x = self.dense2(x)
        x = self.gcn2.call(x, self.adj)

        return self.dense3(x)



