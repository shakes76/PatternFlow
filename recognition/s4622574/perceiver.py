import tensorflow as tf
from tensorflow.keras import layers
from attention import cross_attention_layer
from transformer import transformer_layer
from fourier_encode import FourierEncode
import tensorflow_addons as tfa

"""
Perceiver model, based on the paper by Andrew Jaegle et. al.
"""
class Perceiver(tf.keras.Model):
    def __init__(
        self,
        patch_size,
        data_size,
        latent_size,
        proj_size,
        num_heads,
        num_trans_blocks,
        num_iterations,
        max_freq, 
        num_bands,
        lr,
        epoch,
        weight_decay
    ):
        super(Perceiver, self).__init__()

        self.latent_size = latent_size
        self.data_size = data_size
        self.patch_size = patch_size
        self.proj_size = proj_size
        self.num_heads = num_heads
        self.num_trans_blocks = num_trans_blocks
        self.iterations = num_iterations
        self.max_freq = max_freq
        self.num_bands = num_bands
        self.lr = lr
        self.epoch = epoch
        self.weight_decay = weight_decay

       

    def build(self, input_shape):

        self.latent_array = self.add_weight(
            shape=(self.latent_size, self.proj_size),
            initializer="random_normal",
            trainable=True,
            name='latent'
        )


        self.fourier_encoder = FourierEncode(self.max_freq, self.num_bands)


        self.cross_attention = cross_attention_layer(
            self.latent_size,
            self.data_size,
            self.proj_size,
        )


        self.transformer = transformer_layer(
            self.latent_size,
            self.proj_size,
            self.num_heads,
            self.num_trans_blocks,
        )


        self.global_average_pooling = layers.GlobalAveragePooling1D()


        self.classify = layers.Dense(units=1, activation=tf.nn.sigmoid)


        super(Perceiver, self).build(input_shape)

    def call(self, inputs):
        encoded_imgs = self.fourier_encoder(inputs)
        cross_attention_inputs = [
            tf.expand_dims(self.latent_array, 0),
            encoded_imgs
        ]


        for _ in range(self.iterations):
            latent_array = self.cross_attention(cross_attention_inputs)
            latent_array = self.transformer(latent_array)
            cross_attention_inputs[0] = latent_array


        outputs = self.global_average_pooling(latent_array)


        logits = self.classify(outputs)
        return logits


"""
Training function
"""
def train(model, train_set, val_set, test_set, batch_size):

    X_train, y_train = train_set
    X_train, y_train = X_train[0:len(X_train) // batch_size * batch_size], \
            y_train[0:len(X_train) // batch_size * batch_size]
    
    X_val, y_val = val_set
    X_val, y_val = X_val[0:len(X_val) // batch_size * batch_size], \
            y_val[0:len(X_val) // batch_size * batch_size]

    X_test, y_test = test_set
    X_test, y_test = X_test[0:len(X_test) // batch_size * batch_size], \
            y_test[0:len(X_test) // batch_size * batch_size]


    optimizer = tfa.optimizers.LAMB(
        learning_rate=model.lr, weight_decay_rate=model.weight_decay,
    )


    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
        ],
    )


    history = model.fit(
        X_train, y_train,
        epochs=model.epoch,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        validation_batch_size=batch_size
    )

    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")


    return history