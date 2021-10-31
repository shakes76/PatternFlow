from tensorflow.keras import layers
import tensorflow as tf

def MLP_module(attention, queryDim):
    attention = layers.LayerNormalization()(attention)

    outputs = layers.Dense(queryDim, activation=tf.nn.gelu)(attention)

    return layers.Dense(queryDim)(outputs)

def attention_mechanism(latentDim, inputDim, queryDim):

    normalizeData = layers.Input(shape=(latentDim, queryDim))
    originalSample = layers.Input(shape=(inputDim, queryDim))
    
    latents = layers.LayerNormalization()(normalizeData)
    normalizedInput = layers.LayerNormalization()(originalSample)

    query = layers.Dense(queryDim)(latents)
    key = layers.Dense(queryDim)(normalizedInput)
    value = layers.Dense(queryDim)(normalizedInput)
    
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

    return tf.keras.Model(inputs=[normalizeData, originalSample], outputs=result)