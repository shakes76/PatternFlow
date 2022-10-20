import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


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
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


input_1 = layers.Input(name="pair_x", shape=(224, 224, 1))
input_2 =layers.Input(name="pair_y", shape=(224, 224, 1))

# As mentioned above, Siamese Network share weights between
# tower networks (sister networks). To allow this, we will use
# same embedding network for both tower networks.
tower_1 = embedding_model(input_1)
tower_2 = embedding_model(input_2)

merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
output_layer = layers.Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(1e-3))(merge_layer)
siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
siamese.summary()


class SiameseModel(keras.Model):

    def __init__(self, siamese_network, margin=1.0):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="acc")

    def call(self, inputs):
        return self.siamese_network([inputs[0], inputs[1]])

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            img1, img2, y_true = data
            y_true = tf.expand_dims(y_true, 1)            
            logits = self.siamese_network([img1, img2], training=True)

            loss = keras.losses.binary_crossentropy(y_true, logits)
            acc = keras.metrics.binary_accuracy(y_true, logits)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)

        # Update the accuracy metric.
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    def test_step(self, data):
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
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]


def classifier():
    embedding_model.trainable = False

    model =  tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(224, 224, 1)),
                embedding_model,

                tf.keras.layers.Dense(2, activation='softmax')
            ])
    
    return model


def main():
    embedding_model.summary()
    siamese.summary()

if __name__ == '__main__':
    main()