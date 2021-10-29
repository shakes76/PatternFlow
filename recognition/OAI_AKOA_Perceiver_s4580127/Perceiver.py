import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


""" 
Image Patch extractor to extract a tensor of patches from each image. Taken from keras tutorial.
"""
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    # Returns a tensor of patches from the image
    def call(self, images, *args, **kwargs):
        batch_size = tf.shape(images)[0]
        size = [1, self.patch_size, self.patch_size, 1]
        patches = tf.image.extract_patches(images=images, sizes=size, strides=size, rates=[1, 1, 1, 1], padding="VALID")
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


""" 
Image patch encoder class from keras tutorial. This is used instead of the fourier embedding from the original paper
as it still maintains information about the positional encoding of pixels but is easier to implement.
"""
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_size):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_size)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_size)

    def call(self, patches, *args, **kwargs):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded


class Perceiver:
    def __init__(self, patch_size, data_dim, latent_size, projection_size, num_heads, transformer_layers, dense_units, dropout_rate, depth, classifier_units):
        super(Perceiver, self).__init__()
        self.latent_size = latent_size
        self.data_dim = data_dim
        self.patch_size = patch_size
        self.projection_size = projection_size
        self.num_heads = num_heads
        self.transformer_module_layers = transformer_layers
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.depth = depth
        self.classifier_units = classifier_units

    # Override the keras build method to initialise the required layers in the model
    def build(self, input_shape):
        self.latent_data = self.add_weight(shape=(self.latent_size, self.projection_size), initializer="random_normal", trainable=True)
        self.patch_extractor = Patches(self.patch_size)
        self.patch_encoder = PatchEncoder(self.data_dim, self.projection_size)
        self.cross_attention_module = create_cross_attention_module(self.latent_size, self.data_dim, self.projection_size, self.dense_units, self.dropout_rate)
        self.transformer_module = create_transformer_module(self.latent_size, self.projection_size, self.num_heads, self.transformer_module_layers, self.dense_units, self.dropout_rate)
        self.global_average_pooling = layers.GlobalAveragePooling1D()
        self.dense_classification = create_dense_layers(output_units=self.classifier_units, dropout_rate=self.dropout_rate)

        super(Perceiver, self).build(input_shape)

    # Specifies the flow of data in the network when classifying images
    def call(self, inputs):
        # Extract and encode patches
        patches = self.patch_extractor(inputs)
        encoded_patches = self.patch_encoder(patches)
        cross_attention_inputs = {
            "latent_data": tf.expand_dims(self.latent_data, 0),
            "data_array": encoded_patches,
        }

        # Send our latent data through the cross attention and transformer modules iteratively
        for unused in range(self.depth):
            latent_data = self.cross_attention_module(cross_attention_inputs)
            latent_data = self.transformer_module(latent_data)
            # Set the latent data of the next iteration.
            cross_attention_inputs["latent_data"] = latent_data

        # Final classification through dense layers
        output = self.global_average_pooling(latent_data)
        classification_logits = self.dense_classification(output)
        return classification_logits

    """
    Creates and returns several dense layers followed by a dropout layer
    The number of neurons within each dense layer is specified in an array of integers
    """
    @staticmethod
    def create_dense_layers(output_units, dropout_rate):
        dense_layers = []
        # Loop through, adding each dense layer
        for units in output_units[:-1]:
            dense_layers.append(layers.Dense(units, activation=tf.nn.gelu))
        dense_layers.append(layers.Dense(units=output_units[-1]))
        dense_layers.append(layers.Dropout(dropout_rate))
        dense = keras.Sequential(dense_layers)
        return dense

    """
    Creates and returns a cross attention model which forms a core module of the perceiver.
    The input passes through an attention layer which is added to the latent array input and then normalised.
    This is then passed through several dense layers and the output of this is then added to the attention output
    which is fed in through a skip connection. 
    """
    @staticmethod
    def create_cross_attention_module(latent_size, data_dim, projection_size, dense_units, dropout_rate):
        inputs = {
            "latent_data": layers.Input(shape=(latent_size, projection_size)),
            "data_array": layers.Input(shape=(data_dim, projection_size)),
        }

        # Normalise the latent and data inputs independently
        latent_data = layers.LayerNormalization()(inputs["latent_data"])
        data_array = layers.LayerNormalization()(inputs["data_array"])

        # Create query, key and value vectors through dense layers
        query = layers.Dense(units=projection_size)(latent_data)
        key = layers.Dense(units=projection_size)(data_array)
        value = layers.Dense(units=projection_size)(data_array)

        # Generate cross-attention outputs.
        attention_output = layers.Attention(use_scale=True, dropout=0.1)([query, key, value], return_attention_scores=False)
        # Sum the Latent array obtained through a skip connection to the attention output
        attention_output = layers.Add()([attention_output, latent_data])
        # Apply layer norm.
        attention_output = layers.LayerNormalization()(attention_output)

        # Apply dense layers
        dense = Perceiver.create_dense_layers(dense_units, dropout_rate)
        outputs = dense(attention_output)
        # Sum the dense output with the attention output using a skip connection
        outputs = layers.Add()([outputs, attention_output])

        # Construct into model and return
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    """
    Creates and returns a transformer model which forms a core module of the perceiver.
    Several transformer blocks are constructed, each containing normalization, self attention and a dense component with
    skip connections connecting to the next block. This was modified from the keras transformer tutorial.
    """
    @staticmethod
    def create_transformer_module(latent_size, projection_size, num_heads, transformer_layers, dense_units, dropout_rate):
        inputs = layers.Input(shape=(latent_size, projection_size))
        x0 = inputs
        # Loop through, creating each transformer block
        for unused in range(transformer_layers):
            # Apply layer normalization.
            x1 = layers.LayerNormalization()(x0)
            # Create a multi-head self-attention layer.
            attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_size, dropout=0.1)(x1, x1)
            # Sum the outputs from the skip connection and the self attention layer
            x2 = layers.Add()([attention_output, x0])
            # Apply layer normalization.
            x3 = layers.LayerNormalization()(x2)
            # Go through dense layers
            dense = Perceiver.create_dense_layers(output_units=dense_units, dropout_rate=dropout_rate)
            x3 = dense(x3)
            # Sum the outputs from the skip connection and the dense layers - this will be fed into the next block.
            x0 = layers.Add()([x3, x2])

        # Construct into model and return
        model = keras.Model(inputs=inputs, outputs=x0)
        return model

