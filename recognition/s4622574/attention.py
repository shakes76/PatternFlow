from tensorflow.keras import layers
import tensorflow as tf

def MLP_module(input, queryDim):
    input = layers.LayerNormalization()(input)

    result = layers.Dense(queryDim, activation=tf.nn.gelu)(input)

    return layers.Dense(queryDim)(result)

def attention_mechanism(latentDim, inputDim, queryDim):

    normalizeData = layers.Input(shape=(latentDim, queryDim))
    originalSample = layers.Input(shape=(inputDim, queryDim))
    
    latents = layers.LayerNormalization()(normalizeData)
    normalizedInput = layers.LayerNormalization()(originalSample)

    # query = layers.Dense(queryDim)(latents)
    # key = layers.Dense(queryDim)(normalizedInput)
    # value = layers.Dense(queryDim)(normalizedInput)

    value = projection(queryDim, normalizedInput)
    key = projection(queryDim, normalizedInput)
    query = projection(queryDim, latents)
    
    attention = layers.Attention(use_scale=True)(
        [query, key ,value]
    )
    
    attention = layers.Dense(queryDim)(attention)

    attention = layers.Add()([attention, latents])

    # attention = layers.LayerNormalization()(attention)

    # outputs = layers.Dense(queryDim, activation=tf.nn.gelu)(attention)

    # outputs = layers.Dense(queryDim)(outputs)

    result = MLP_module(attention, queryDim)

    result = layers.Add()([result, attention])

    model = tf.keras.Model(inputs=[normalizeData, originalSample], outputs=result)

    return model

def projection(queryDim, input):
    output = layers.Dense(queryDim)(input)
    return output