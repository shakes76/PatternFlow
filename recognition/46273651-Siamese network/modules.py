import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np



"""
A CNN module that used to extract the feature from the input images.
"""
embedding_model = tf.keras.Sequential(
    [
        keras.Input(shape=(224, 224, 1)),
        layers.Conv2D(64, (10,10), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (7,7), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        layers.MaxPooling2D(pool_size=(2, 2)),


        layers.Conv2D(128, (4,4), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, (4,4), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3)),

        layers.Flatten(),
        layers.Dense(256, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
    ],
    name="embedding"
)

embedding_model.summary()

# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: Input1 and input2.

    Returns:
        A vector containing the Euclidean distance between these two vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


"""
A Siamese network that takes 2 images as input and learns a shared embedding across them.
The output of the network is a number between [0,1] based on euclidian distances of the input.
"""
input_1 = layers.Input(name="pair_x", shape=(224, 224, 1))
input_2 =layers.Input(name="pair_y", shape=(224, 224, 1))

tower_1 = embedding_model(input_1)
tower_2 = embedding_model(input_2)

merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
output_layer = layers.Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(1e-3))(merge_layer)
siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
siamese.summary()


class SiameseModel(keras.Model):
    """
    A Siamese Model that used to train the siamese network.
    The loss function used to train the model is binary_cross_entropy
    The accuracy metric is binary_accuracy
    """

    def __init__(self, siamese_network = siamese):
        """
        A Keras Model Wrapper to train the siames network.

        Args:
            siamese_network: A siamese network model that used to determine the similarity of 2 input images.
            loss_tracker: calculate the loss of the model
            acc_tracker: calculate accuracy of the model
        """
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="acc")

    def call(self, inputs):
        """
        Returns the output of the network as provided input to the network.

        Args:
            inputs: A input Tensor.

        Returns:
            the output of the network.
        """
        return self.siamese_network([inputs[0], inputs[1]])

    def train_step(self, data):
        """
        The training step is used to update the weights of the network.

        Args:
            data: input pairs with their labels.
        Returns:
            The loss and accuracy of single step of the network.
        """

        with tf.GradientTape() as tape:
            img1, img2, y_true = data
            y_true = tf.expand_dims(y_true, 1)            
            logits = self.siamese_network([img1, img2], training=True)

            loss = keras.losses.binary_crossentropy(y_true, logits)
            acc = keras.metrics.binary_accuracy(y_true, logits)

        # Storing the gradients of the loss function
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # update and return the training loss metric.
        self.loss_tracker.update_state(loss)

        # Update the accuracy metric.
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    def test_step(self, data):
        """
        The test step is used to test performace of the network.

        Args:
            data: input pairs with their labels.
        Returns:
            The loss and accuracy of single step of the network.
        """

        img1, img2, y_true = data
        y_true = tf.expand_dims(y_true, 1)

        loss = keras.losses.binary_crossentropy(y_true, self.siamese_network([img1, img2]))
        acc = keras.metrics.binary_accuracy(y_true, self.siamese_network([img1, img2]))

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)

        # Update the accuracy metric.
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]


def classifier():
    embedding_model.trainable = False

    model =  tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(224, 224, 1)),
                embedding_model,

                tf.keras.layers.Dense(2, activation='softmax')
            ])
    
    return model
