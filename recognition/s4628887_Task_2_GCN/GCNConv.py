import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers, activations, initializers, constraints, backend


class GCNConv(Layer):


    def __init__(self,
               channels,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               **kwargs):
        super(GCNConv, self).__init__(
        trainable=trainable,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
        self.channels = channels
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)


    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.built = True

    def call(self, inputs, mask=None):
        x, a = inputs

        output = backend.dot(x, self.kernel)
        output = backend.dot(a, output)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if mask is not None:
            output *= mask[0]
        output = self.activation(output)

        return output

    @property
    def config(self):
        return {"channels": self.channels}

