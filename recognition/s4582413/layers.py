from tensorflow.keras import layers
import tensorflow as tf

# Create extra layers/component needed for the perceiver transformer architecture
LATENT_SIZE = 256
PROJ_SIZE = 27
LATENT_SHAPE = (LATENT_SIZE, PROJ_SIZE)
NUM_HEADS = 8


# Gets the component that makes up the attention layer
# returns the attention layer and projection layer respecitvely
def get_attention_comp(inputs, norm_input):
    # create the attention layer followed by linear layer and normalisation
    layer = layers.MultiHeadAttention(NUM_HEADS, PROJ_SIZE)(norm_input, norm_input)
    layer = layers.Dense(PROJ_SIZE)(layer)
    layer = layers.Add()([layer, inputs])
    layer = layers.LayerNormalization()(layer)

    projection_layer = layers.Dense(PROJ_SIZE, activation=tf.nn.gelu)(layer)

    return layer, projection_layer


# Gets the transformer layer needed for this task
def get_transformer_layer():
    inputs = layers.Input(LATENT_SHAPE)
    normalised_inputs = layers.LayerNormalization()(inputs)

    # 6 transformer block as described in the paper
    for i in range(0, 6):
        # obtain the attention component for the transformer layer and adding the layers up
        attention_component = get_attention_comp(inputs, normalised_inputs)
        # Applying transformation on the attention result
        attention_transformation = layers.Add()([layers.Dense(PROJ_SIZE)(attention_component[1]), attention_component[0]])

    # return the build model
    return tf.keras.Model(inputs=inputs, outputs=attention_transformation)