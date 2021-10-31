from tensorflow.keras import layers
import tensorflow as tf 

def attention_mechanism(latentDim, inputDim, queryDim):

    normalizeData = layers.Input((latentDim, queryDim))
    latents = layers.LayerNormalization()(normalizeData)

    originalSample = layers.Input((inputDim, queryDim))
    normalizedInput = layers.LayerNormalization()(originalSample)

    query = layers.Dense(queryDim)(latents)
    key = layers.Dense(queryDim)(normalizedInput)
    value = layers.Dense(queryDim)(normalizedInput)
    
    attention = layers.Attention(use_scale=True)([query, key ,value])
    
    attention = layers.Dense(queryDim)(attention)

    attention = layers.Add()([attention, latents])

    attention = layers.LayerNormalization()(attention)

    outputs = layers.Dense(queryDim, activation=tf.nn.gelu)(attention)

    outputs = layers.Dense(queryDim)(outputs)

    outputs = layers.Add()([outputs, attention])

    return tf.keras.Model(inputs=[normalizeData, originalSample], outputs=outputs)