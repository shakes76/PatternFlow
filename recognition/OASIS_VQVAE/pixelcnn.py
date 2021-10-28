import tensorflow as tf

class PixelConvLayer(tf.keras.layers.Layer):
    """
    A masked convolutional layer, as described in the PixelCNN paper.
    """

    def __init__(self, mask_type, **kwargs):
        """
        Parameters:
            mask_type (str): "A" or "B" to indicate mask type.
        """
        super(PixelConvLayer, self).__init__()
        self._mask_type = mask_type
        self._conv = tf.keras.layers.Conv2D(**kwargs)

    def build(self, input_shape):
        self._conv.build(input_shape)

        kernel_shape = self._conv.kernel.get_shape()

        # the below is not great, but the only way I could find to do this in tensorflow :(
        mask = tf.Variable(tf.zeros(shape=kernel_shape))

        mask[: kernel_shape[0] // 2, ...].assign(
            tf.ones(tf.shape(mask[: kernel_shape[0] // 2, ...])))

        mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...].assign(
            tf.ones(tf.shape(mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...])))

        if self._mask_type == "B":
            mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...].assign(
                tf.ones(tf.shape(mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...])))

        self._mask = tf.convert_to_tensor(mask)

    def call(self, inputs):
        self._conv.kernel.assign(self._conv.kernel * self._mask)
        return self._conv(inputs)


class ResidualBlock(tf.keras.layers.Layer):
    """
    Residual block that uses a PixelConvLayer
    """

    def __init__(self, filters, **kwargs):
        """
        Parameters:
            filters (int): number of filters to use
        """

        super(ResidualBlock, self).__init__(**kwargs)

        self._conv1 = tf.keras.layers.Conv2D(filters, kernel_size=1, activation="relu")

        self._pixel_conv = PixelConvLayer(mask_type="B",
                                         filters=filters // 2,
                                         kernel_size=3,
                                         activation="relu",
                                         padding="same")

        self._conv2 = tf.keras.layers.Conv2D(filters, kernel_size=1, activation="relu")

    def call(self, inputs):
        x = self._conv1(inputs)
        x = self._pixel_conv(x)
        x = self._conv2(x)
        return tf.keras.layers.add([inputs, x])


class PixelCNN(tf.keras.Model):
    """
    The overall PixelCNN model
    """

    def __init__(self, 
                 input_shape, 
                 num_embeddings, 
                 filters, 
                 res_blocks, 
                 pixelcnn_layers, 
                 name="pixelcnn", **kwargs):
        """
        Parameters:
            input_shape (int, int): The shape (not including batch dimension) of the codebook
            num_embeddings (int): Number of embeddings in the trained codebook.
            filters (int): Number of filters to use in each PixelConvLayer
            res_blocks (int): Number of ResidualBlock to use in the PixelCNN
            pixelcnn_layers (int): Number of additional middle PixelConvLayers to add.
        """

        super(PixelCNN, self).__init__(name=name, **kwargs)

        layers = []

        # Input layer
        layers.append(tf.keras.layers.InputLayer(input_shape=input_shape, dtype=tf.int32))

        # One hot encoding layer
        layers.append(tf.keras.layers.Lambda(lambda x: tf.one_hot(x, num_embeddings)))

        layers.append(PixelConvLayer(
            mask_type="A", filters=filters, kernel_size=7, activation="relu", padding="same"
        ))

        # add the ResidualBlocks
        layers.extend([ResidualBlock(filters) for i in range(res_blocks)])

        # add the PixelConvLayers
        for i in range(pixelcnn_layers):
            layers.append(PixelConvLayer(
                mask_type="B",
                filters=filters,
                kernel_size=1,
                strides=1,
                activation="relu",
                padding="valid",
            ))

        layers.append(keras.layers.Conv2D(filters=num_embeddings, kernel_size=1, strides=1, padding="valid"))

        # pipe the layers together
        self._pixel_network = tf.keras.Sequential(layers, name="pixelcnn_net")

        self._total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    def call(self, x):
        return self.pixel_network(x)

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Calculate the loss
            predictions = self(x)
            loss = self.compiled_loss(x, predictions)

        # Backpropagate gradients
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self._total_loss_tracker.update_state(loss)
        return {"loss": self._total_loss_tracker.result()}

