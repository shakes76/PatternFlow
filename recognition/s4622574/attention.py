from tensorflow.keras import layers
import tensorflow as tf

def MLP_module(input, queryDim):
    """Feedforward the input to MLP"""
    #use GELU
    input = layers.LayerNormalization()(input)
    result = layers.Dense(queryDim, activation=tf.nn.gelu)(input)  
    return layers.Dense(queryDim)(result)

def attention_mechanism(latentDim, inputDim, queryDim):
    """Attention Mechanism:
    Return: Attention Layers having latent and patient picture
    with corresponding output
    """
    # To be normalized 
    normalizeData = layers.Input(shape=(latentDim, queryDim))
    # To be normalized 
    originalSample = layers.Input(shape=(inputDim, queryDim))
    latents = layers.LayerNormalization()(normalizeData)

    normalizedInput = layers.LayerNormalization()(originalSample)

    # Query -> Key, Value - Attention Mechanism
    value = projection(queryDim, normalizedInput)

    key = projection(queryDim, normalizedInput)

    query = projection(queryDim, latents)
    isScaled = True
    att = layers.Attention(use_scale=isScaled)(
        [query, key ,value]
    )
    att = layers.Dense(queryDim)(att)

    att = layers.Add()([att, latents])
    result = MLP_module(att, queryDim) #Feedforward to fully connected layers
    result = layers.Add()([result, att])

    model = tf.keras.Model(outputs=result, inputs=[normalizeData, originalSample])
    return model

def projection(queryDim, input):
    """Projection of input with corresponding dimension"""
    output = layers.Dense(queryDim)(input)
    return output