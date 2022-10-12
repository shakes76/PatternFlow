"""
Assumptions:

Steps / Key Functions:
1. Augment Data (data_augmentation)
2. Create Patches (Patches)
3. Embed Patches (PatchEmbedding)
4. Create MLP (mlp)
5. Create Transformer Encoder (transformer_encoder)
5. Create ViT (vit_classifier)

References:
1) https://keras.io/examples/vision/image_classification_with_vision_transformer/
2) https://towardsdatascience.com/understand-and-implement-vision-transformer-with-tensorflow-2-0-f5435769093

"""


from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

##############################  INPUT DATA AUGMENTATION  ###################################

def data_augmentation(mean, variance):
    """ data augmentation for input data based on calculated mean and variance of training data """

    data_augmentation = keras.Sequential(
        [
            layers.Normalization(mean=mean, variance=variance),
        ],
        name="data_augmentation",
    )

    return data_augmentation


###################################  CREATE PATCHES  #######################################

class Patches(layers.Layer):
    """ Class to create patches from input images"""
    def __init__(self, patch_size):
        """ Constructor calling Layers first"""
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        """ Allows Patches class to act like a method with images as input """
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

###################################  EMBED PATCHES  #######################################

###################################  EMBED PATCHES  #######################################

class PatchEmbedding(layers.Layer):
    """ Class to linear project flattened patch into projection_dim and add positional embedding and class token"""
    def __init__(self, num_patches, projection_dim):
        super(PatchEmbedding, self).__init__()
        
        # linear projection onto projection dims
        self.projection = layers.Dense(units=projection_dim)
        
        # position embedding
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
        
        # weight initializer for class token
        class_weights_init = tf.random_normal_initializer()
        
        # initialize class token with initial values from weight initializer
        class_token = class_weights_init(shape=(1, projection_dim), dtype="float32")
        self.class_token = tf.Variable(initial_value=class_token, trainable=True)
        
    def call(self, patch):
        # reshape class token to take into account batch size and projection dims
        class_token = tf.tile(input=self.class_token, multiples=[BATCH_SIZE, 1])
        class_token = tf.reshape(class_token, (BATCH_SIZE, 1, projection_dim))
        
        # create linear positions (one for each patch) and add one for class token
        positions = tf.range(start=0, limit=self.num_patches+1, delta=1)

        # create position embedding
        position_embedding = self.position_embedding(positions)
        
        # project patch into projection dims
        patch_embedding = self.projection(patch)
        
        # prepend class token
        patch_embedding = tf.concat([class_token, patch_embedding], axis=1)
        
        # embedding is patch embedding plus position embedding
        embedding = patch_embedding + position_embedding
        
        return embedding


###################################  CREATE MLP  #######################################

def mlp(x, hidden_units, dropout_rate):
    """ Generic function to create zero or more mlp blocks each a dense layer and a dropout layer  """
    
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.keras.activations.tanh)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


##############################  CREATE TRANSFORMER ENCODER  ##################################

def transformer_encoder(embedded_patches, num_encoder_layers, dropouts, projection_dim):
    """ Create transformer encoder block """
    
    # extract dropouts
    mha_dropout = dropouts["mha"]
    mlp_dropout = dropouts["encoder_mlp"]
    
    # so that multiple encoder layers can be generated from embeddedings
    encoded_patches = embedded_patches
    
    # create one or more layers
    for _ in range(num_encoder_layers):
        
        # normalization lyaer
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # multi-head self-attention layer
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
        x2 = layers.MultiHeadAttention(num_heads=num_heads, 
                                       key_dim=projection_dim, 
                                       dropout=mha_dropout)(x1, x1)
        
        # residual connection
        x3 = layers.Add()([x2, encoded_patches])
        
        # normalization layer
        x4 = layers.LayerNormalization(epsilon=1e-6)(x3)
        
        # MLP.
        hidden_units = [projection_dim * 2, projection_dim]
        x5 = mlp(x4, hidden_units=hidden_units, dropout_rate=mlp_dropout)
        
        # residual connection
        encoded_patches = layers.Add()([x5, x3])
        
    return encoded_patches


##############################  CREATE VISION TRANSFORMER MODEL  #################################

def vit_classifier():
    
    inputs = layers.Input(shape=input_shape)
    
    # Augment data.
    augmented = data_augmentation(mean=mean, variance=variance)(inputs)
    
    # Create patches.
    patches = Patches(patch_size)(augmented)
    
    # create patch embeddings
    embedded_patches = PatchEmbedding(num_patches, projection_dim)(patches)

    # create patch encodings
    encoded_patches = transformer_encoder(embedded_patches, num_encoder_layers, dropouts, projection_dim)

    # prepare patch encodings for mlp
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.2)(representation)
    
#     representation = tf.reduce_mean(encoded_patches, axis=1)
    
    # MLP head
    features = mlp(x=representation, hidden_units=[2048], dropout_rate=0.5)
    
    # Classify outputs.
    outputs = layers.Dense(num_classes)(features)
#     outputs = layers.Dense(num_classes)(representation)
    
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model