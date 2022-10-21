import tensorflow as tf
from keras import layers

##### generate patches 
class generate_patch(layers.Layer):
    def __init__(self, patch_size):
        super(generate_patch, self).__init__()
        self.patch_size = patch_size
    
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images, sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding="VALID")
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims]) #here shape is (batch_size, num_patches, patch_h*patch_w*c) 
        return patches

### Positonal Encoding Layer
class PatchEncode_Embed(layers.Layer):
    '''
    2 steps happen here
    1. flatten the patches
    2. Map to dim D; patch embeddings
    '''
    def __init__(self, num_patches, projection_dim):
        super(PatchEncode_Embed, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim)
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        
        return encoded

'''
This part takes images as inputs,
Conv layer filter matches query dim of multi-head attention layer 
Add embeddings by randomly initializing the weights
'''
def generate_patch_conv_orgPaper_f(patch_size, hidden_size, inputs):
    patches = layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    row_axis, col_axis = (1, 2) # channels last images
    seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
    x = tf.reshape(patches, [-1, seq_len, hidden_size])
    return x

### Positonal Encoding Layer
class AddPositionEmbs(layers.Layer):

    """inputs are image patches 
    Custom layer to add positional embeddings to the inputs."""
    def __init__(self, posemb_init=None, **kwargs):
        super().__init__(**kwargs)
        self.posemb_init = posemb_init
        #posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input') # used in original code

    def build(self, inputs_shape):
        pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
        self.pos_embedding = self.add_weight('pos_embedding', pos_emb_shape, initializer=self.posemb_init)

    def call(self, inputs, inputs_positions=None):
        # inputs.shape is (batch_size, seq_len, emb_dim).
        pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)

        return inputs + pos_embedding

'''
part of ViT Implementation
this block implements the Transformer Encoder Block
Contains 3 parts--
1. LayerNorm 2. Multi-Layer Perceptron 3. Multi-Head Attention
For repeating the Transformer Encoder Block we use Encoder_f function. 
'''
def mlp_block_f(mlp_dim, dropout, inputs):
    x = layers.Dense(units=mlp_dim, activation=tf.nn.gelu)(inputs)
    x = layers.Dropout(rate=dropout)(x) # dropout rate is from original paper,
    x = layers.Dense(units=inputs.shape[-1], activation=tf.nn.gelu)(x) # check GELU paper
    x = layers.Dropout(rate=dropout)(x)
    return x

def Encoder1Dblock_f(num_heads, mlp_dim, dropout, inputs):
    x = layers.LayerNormalization(dtype=inputs.dtype)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=dropout)(x, x) 
    # self attention multi-head, dropout_rate is from original implementation
    x = layers.Add()([x, inputs]) # 1st residual part 

    y = layers.LayerNormalization(dtype=x.dtype)(x)
    y = mlp_block_f(mlp_dim, dropout, y)
    y_1 = layers.Add()([y, x]) #2nd residual part 
    return y_1

def Encoder_f(num_layers, mlp_dim, num_heads, dropout, emb_dropout, inputs):
    x = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input')(inputs)
    x = layers.Dropout(rate=emb_dropout)(x)
    for _ in range(num_layers):
        x = Encoder1Dblock_f(num_heads, mlp_dim, dropout, x)

    encoded = layers.LayerNormalization(name='encoder_norm')(x)
    return encoded

'''
Building blocks of ViT
Check other gists or the complete notebook
[]
Patches (generate_patch_conv_orgPaper_f) + embeddings (within Encoder_f)
Transformer Encoder Block (Encoder_f)
Final Classification 
'''
def build_ViT(preprocessing, image_size, transformer_layers, patch_size, hidden_size, num_heads, mlp_dim, num_classes, dropout, emb_dropout):
    inputs = layers.Input(shape=(image_size[0], image_size[1], 1))
    preprocessing = preprocessing(inputs)

    # generate patches with conv layer
    patches = generate_patch_conv_orgPaper_f(patch_size, hidden_size, preprocessing)

    ######################################
    # ready for the transformer blocks
    ######################################
    encoder_out = Encoder_f(transformer_layers, mlp_dim, num_heads, dropout, emb_dropout, patches)  

    #####################################
    #  final part (mlp to classification)
    #####################################
    #encoder_out_rank = int(tf.experimental.numpy.ndim(encoder_out))
    im_representation = tf.reduce_mean(encoder_out, axis=1)  # (1,) or (1,2)
    # similar to the GAP, this is from original Google GitHub

    logits = layers.Dense(units=num_classes, name='head', kernel_initializer=tf.keras.initializers.zeros)(im_representation)
    # !!! important !!! activation is linear 

    final_model = tf.keras.Model(inputs = inputs, outputs = logits)
    return final_model