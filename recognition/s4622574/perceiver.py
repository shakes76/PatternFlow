import tensorflow as tf
from tensorflow.keras import layers
from attention import attention_mechanism
from fourier_encode import FourierEncode
import tensorflow_addons as tfa
import copy

class Perceiver(tf.keras.Model):
    def __init__(self, inDim, latentDim, proj_size, num_heads,
            num_trans_blocks, num_loop, max_freq, freq_ban, lr, epoch, 
            weight_decay):

        super(Perceiver, self).__init__()

        self.latentDim = latentDim
        self.inDim = inDim
        self.proj_size = proj_size
        self.num_heads = num_heads
        self.num_trans_blocks = num_trans_blocks
        self.loop = num_loop
        self.max_freq = max_freq
        self.freq_ban = freq_ban
        self.lr = lr
        self.epoch = epoch
        self.weight_decay = weight_decay

    def build(self, input_shape):

        self.latents = self.add_weight(shape=(self.latentDim, self.proj_size),
                initializer="random_normal", trainable=True, name='latent')

        self.fourier_encoder = FourierEncode(self.max_freq, self.freq_ban)


        self.attention_mechanism = attention_mechanism(self.latentDim,
                self.inDim, self.proj_size)

        self.transformer = transform(self.latentDim, self.proj_size,
                self.num_heads, self.num_trans_blocks)

        self.global_average_pooling = layers.GlobalAveragePooling1D()

        self.classify = layers.Dense(units=1, activation=tf.nn.sigmoid)

        super(Perceiver, self).build(input_shape)

    def call(self, inputs):
        fourier_transform = self.fourier_encoder(inputs)
        attention_mechanism_data = [tf.expand_dims(self.latents, 0), fourier_transform]


        for _ in range(self.loop):
            latents = self.attention_mechanism(attention_mechanism_data)
            latents = self.transformer(latents)
            attention_mechanism_data[0] = latents


        outputs = self.global_average_pooling(latents)


        logits = self.classify(outputs)
        return logits


def fitModel(model, train_set, val_set, test_set, batch_size):

    X_train, y_train = train_set
    X_train, y_train = X_train[0:len(X_train) // 32 * 32], y_train[0:len(X_train) // 32 * 32]
    
    X_val, y_val = val_set
    X_val, y_val = X_val[0:len(X_val) // 32 * 32], y_val[0:len(X_val) // 32 * 32]

    X_test, y_test = test_set
    X_test, y_test = X_test[0:len(X_test) // 32 * 32], y_test[0:len(X_test) // 32 * 32]


    optimizer = tfa.optimizers.LAMB(
        learning_rate=model.lr, weight_decay_rate=model.weight_decay,
    )


    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),])


    history = model.fit(X_train, y_train, epochs=model.epoch, batch_size=batch_size, 
            validation_data=(X_val, y_val), validation_batch_size=batch_size)

    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history

def transform(latentDim, proj_size, num_heads, num_trans_blocks):
    data = layers.Input(shape=(latentDim, proj_size))
    originalInput = copy.deepcopy(data)
    for _ in range(num_trans_blocks):
        norm = layers.LayerNormalization()(data)
        attOut = layers.MultiHeadattOut(num_heads, proj_size)(norm, norm)
        attOut = layers.Dense(proj_size)(attOut)
        attOut = layers.Add()([attOut, data])
        attOut = layers.LayerNormalization()(attOut)
        outputs = layers.Dense(proj_size, activation=tf.nn.gelu)(attOut)
        outputs = layers.Dense(proj_size)(outputs)
        finalOut = layers.Add()([outputs, attOut])
    return tf.keras.Model(inputs=data, outputs=finalOut)