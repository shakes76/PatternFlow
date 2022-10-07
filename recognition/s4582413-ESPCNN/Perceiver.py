import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow_addons as tfa
import tensorflow.keras as keras
from tensorflow.keras import layers

from CrossAttention import cross_attention
from FourierEncode import fourier_encode
from layers import get_transformer_layer


class Perceiver(tf.keras.Model):
    """
    Creates the Perceiver transformer model as described in the paper. 
    """

    def __init__(
            self,
            data_size,  # size of data
            latent_size,  # size of latent dimension
            proj_size,  # projection size of the fourier feature
            num_heads,  # number of heads in the mutli-head attention layer of the transformer
            num_transformer_blocks,  # Number of transformer block in the model
            num_iterations,  # Number of iteration to apply for the cross-attention and transformer layer
            max_freq,  # The Nyquist frequency of the Fourier feature
            num_bands,  # The number of frequency bands in the fourier feature
            learning_rate,  # Learning rate for the optimiser
            epoch,  # Number of epoch for the training
            weight_decay  # The decay weight for the optimiser
    ):
        super(Perceiver, self).__init__()
        self.latent_size = latent_size
        self.data_size = data_size
        self.proj_size = proj_size
        self.num_heads = num_heads
        self.num_trans_blocks = num_transformer_blocks
        self.iterations = num_iterations
        self.max_freq = max_freq
        self.num_bands = num_bands
        self.lr = learning_rate
        self.epoch = epoch
        self.weight_decay = weight_decay

    """
    Override the build model in the Model class
    Params:
        input_dim: Input dimension as specified
    """

    def build(self, input_dim):
        # Create components needed for the model,
        # the latent array, cross-attention module, and the usual transformer architecture
        # (consist of transformer layer, average pooling layer and classification layer)

        # latent array
        self.latent_array = self.add_weight(
            shape=(self.latent_size, self.proj_size),
            initializer="random_normal",
            trainable=True,
            name='latent'
        )

        # cross-attention module
        self.cross_attention = cross_attention(self.data_size)

        # Transformer layer
        self.transformer_layer = get_transformer_layer()

        # Global average pooling layer
        self._global_avg_layer = layers.GlobalAveragePooling1D()

        # classification layer
        self.final_classify = layers.Dense(units=1, activation=tf.nn.sigmoid)

        # Now have all the components required for the perceiver transformer architecture, and can build
        # the model
        super(Perceiver, self).build(input_dim)

    """
    Overrides the call() method in Model class. 
    Returns an array containing the prediction of the model given the image data
    """

    def call(self, inputs):
        # Gets the cross-attention inputs for the image
        encoded_input = fourier_encode(inputs)
        attention_inputs = [
            tf.expand_dims(self.latent_array, 0),
            encoded_input
        ]

        # Apply iterative cross-attention and transformer layer transformation
        for i in range(self.iterations):
            latent_data = self.cross_attention(attention_inputs)
            latent_data = self.transformer_layer(latent_data)
            attention_inputs[0] = latent_data

        # Now computing the global average pooling in the transformer architecture
        raw_output = self.global_average_pooling(latent_data)
        # generate the predicted result
        prediction = self.final_classify(raw_output)

        return prediction

    """
        Training function for the perceiver model
        Contains the training, validation, test data set and batch size for training
    """

    def train(self, train_data, val_data, test_data, batch_size):
        # Obtain the training, validation and test data set for training based on the batch_size

        # data of index 0 represents X, the input data while 1 represent the predicted result
        X_train, y_train = train_data
        X_train, y_train = train_data[0][0:len(train_data[0]) // batch_size * batch_size], \
                                       train_data[1][0:len(train_data[0]) // batch_size * batch_size]

        # repeat for validation and test set
        X_val, y_val = val_data
        X_val, y_val = val_data[0][0:len(val_data[0]) // batch_size * batch_size], \
                                   val_data[1][0:len(val_data[0]) // batch_size * batch_size]

        X_test, y_test = test_data
        X_test, y_test = test_data[0][0:len(test_data[0]) // batch_size * batch_size], \
                                     test_data[1][0:len(test_data[0]) // batch_size * batch_size]

        # Choose to use the LAMB optimiser as described used in the paper
        optimizer = tfa.optimizers.LAMB(
            learning_rate=self.lr, weight_decay_rate=self.weight_decay,
        )

        # Fit the model
        model_history = self.fit(
            X_train, y_train,
            epochs= self.epoch,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            validation_batch_size=batch_size
        )

        _, model_accuracy = self.evaluate(X_test, y_test)

        print(f"Test accuracy: {round(model_accuracy * 100, 2)}%")

        # for visualising the learning curve of the model
        return model_history











