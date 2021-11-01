import tensorflow as tf
from tensorflow.keras import layers
from attention import attention_mechanism
from fourier_encode import FourierEncode
import tensorflow_addons as tfa
import copy

class Perceiver(tf.keras.Model):
    def __init__(self, inDim, latentDim, queryDim, max_freq, freq_ban, head=8,
            block=6, num_loop=8,  learningRate=0.001, epoch=10, 
            decayRate=0.0001):

        super(Perceiver, self).__init__()

        self.latentDim = latentDim
        self.inDim = inDim
        self.queryDim = queryDim
        self.head = head
        self.block = block
        self.loop = num_loop
        self.max_freq = max_freq
        self.freq_ban = freq_ban
        self.learningRate = learningRate
        self.epoch = epoch
        self.decayRate = decayRate

    def build(self, input_shape):

        self.latents = self.add_weight(shape=(self.latentDim, self.queryDim),
                initializer="random_normal", trainable=True, name='latent')

        self.fourier_encoder = FourierEncode(self.max_freq, self.freq_ban)

        self.attention_mechanism = attention_mechanism(self.latentDim, self.inDim, self.queryDim)

        self.transformer = transform(self.latentDim, self.queryDim, self.head, self.block)

        self.global_average_pooling = layers.GlobalAveragePooling1D()

        self.classify = layers.Dense(units=1, activation=tf.nn.sigmoid)

        super(Perceiver, self).build(input_shape)

    def call(self, inputs):
        """Augmentation"""

        fourier_transform = self.fourier_encoder(inputs)

        attention_mechanism_data = [tf.expand_dims(self.latents, 0), fourier_transform]
        for tempVar in range(self.loop):
            latents = self.attention_mechanism(attention_mechanism_data)
            latents = self.transformer(latents)
            attention_mechanism_data[0] = latents
        outputs = self.global_average_pooling(latents)
        logits = self.classify(outputs)

        return logits


def fitModel(model, training, validation, testing, numSamples):
    """Compile and Train"""
    testX, testY = testing
    testX, testY = testX[0:len(testX) // 32 * 32], testY[0:len(testX) // 32 * 32]
    trainX, trainY = training
    trainX, trainY = trainX[0:len(trainX) // 32 * 32], trainY[0:len(trainX) // 32 * 32]   
    valX, valY = validation
    valX, valY = valX[0:len(valX) // 32 * 32], valY[0:len(valX) // 32 * 32]
    optimizer = tfa.optimizers.LAMB(
        learning_rate=model.learningRate, weight_decay_rate=model.decayRate,
    )
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),])
    history = model.fit(trainX, trainY, epochs=model.epoch, batch_size=numSamples, 
            validation_data=(valX, valY), validation_batch_size=numSamples)
    varTemp, accuracy = model.evaluate(testX, testY)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    return history

def transform(latentDim, queryDim, head, block):
    data = layers.Input(shape=(latentDim, queryDim))
    originalInput = copy.deepcopy(data)
    for _ in range(block):
        norm = layers.LayerNormalization()(data)
        attOut = layers.MultiHeadAttention(head, queryDim)(norm, norm)
        attOut = layers.Dense(queryDim)(attOut)
        attOut = layers.Add()([attOut, data])
        attOut = layers.LayerNormalization()(attOut)
        outputs = layers.Dense(queryDim, activation=tf.nn.gelu)(attOut)
        outputs = layers.Dense(queryDim)(outputs)
        finalOut = layers.Add()([outputs, attOut])
    return tf.keras.Model(inputs=data, outputs=finalOut)