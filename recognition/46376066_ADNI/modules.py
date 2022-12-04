import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras import layers

# Perceiver Architecture

def get_forward(hidden_units, dropout_rate):
    forward_layers = []
    for units in hidden_units[:-1]:
        forward_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    forward_layers.append(layers.Dense(units=hidden_units[-1]))
    forward_layers.append(layers.Dropout(dropout_rate))

    ffn = keras.Sequential(forward_layers)
    
    return ffn

## Cross-Attention

def get_x_attention_mod(latent_dims,
    data_dims,
    proj_dims,
    forward_units,
    dropout_rate
):
    latent_arr = layers.Input(shape=(latent_dims, proj_dims))
    data_arr = layers.Input(shape=(data_dims, proj_dims))

    # Latent
    query = layers.Dense(units=proj_dims)(latent_arr)
    # Data
    key = layers.Dense(units=proj_dims)(data_arr)
    value = layers.Dense(units=proj_dims)(data_arr)

    attention_output = layers.Attention(use_scale=True, dropout=0.1)(
        [query, key, value], return_attention_scores=False
    )

    attention_output = layers.Add()([attention_output, latent_arr])

    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)

    f_net = get_forward(hidden_units=forward_units, dropout_rate=dropout_rate)
    outputs = f_net(attention_output)

    outputs = layers.Add()([outputs, attention_output])

    #If something goes wrong iiput methods will be probelm
    model = keras.Model(inputs=[latent_arr, data_arr], outputs=outputs)

    return model


## Transformer

def get_transformer_mod(latent_dim,
    proj_dim,
    num_heads,
    num_t_blocks,
    forward_units,
    dropout_rate,
):

    inputs = layers.Input(shape=(latent_dim, proj_dim))

    x_0 = inputs
    
    for _ in range(num_t_blocks):
        
        x_1 = layers.LayerNormalization(epsilon=1e-6)(x_0)
        
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=proj_dim, dropout=0.1
        )(x_1, x_1)
        
        x_2 = layers.Add()([attention_output, x_0])
        
        x_3 = layers.LayerNormalization(epsilon=1e-6)(x_2)

        f_net = get_forward(hidden_units=forward_units, dropout_rate=dropout_rate)
        x_3 = f_net(x_3)

        x_0 = layers.Add()([x_3, x_2])

    model = keras.Model(inputs=inputs, outputs=x_0)
    return model

## Iterative cross-attention and weight sharing?? or Perciever is alternating of above 2. need to check

## Perceiver Model

class Perceiver(keras.Model):
    def __init__(
        self,
        patch_size,
        data_dim,
        latent_dim,
        proj_dim,
        num_heads,
        num_t_blocks,
        forward_units,
        dropout_rate,
        num_iters,
        classif_units
    ):
        super(Perceiver, self).__init__()
        self.patch_size = patch_size
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.num_t_blocks = num_t_blocks
        self.forward_units = forward_units
        self.dropout_rate = dropout_rate
        self.num_iters = num_iters
        self.classif_units = classif_units

    def build(self, input_shape):
        self.latent_arr = self.add_weight(
            shape=(self.latent_dim, self.proj_dim),
            initializer="random_normal",
            trainable=True,
            name='name'
        )

        self.patcher = PatchCreater(self.patch_size)

        self.patch_pos_encoder = PatchPosEncoder(self.data_dim, self.proj_dim)

        self.x_attention = get_x_attention_mod(
            self.latent_dim,
            self.data_dim,
            self.proj_dim,
            self.forward_units,
            self.dropout_rate
        )

        self.transformer = get_transformer_mod(
            self.latent_dim,
            self.proj_dim,
            self.num_heads,
            self.num_t_blocks,
            self.forward_units,
            self.dropout_rate
        )

        self.avg_pooling = layers.GlobalAveragePooling1D()

        self.classif_head = get_forward(
            hidden_units=self.classif_units, dropout_rate= self.dropout_rate
        )

        super(Perceiver, self).build(input_shape)

    def call(self, inputs):
        patches = self.patcher(inputs)

        pos_encode_patches = self.patch_pos_encoder(patches)

        latent_arr = tf.expand_dims(self.latent_arr, 0)
        data_arr = pos_encode_patches

        for _ in range(self.num_iters):
            latent_arr = self.x_attention([latent_arr, data_arr])
            latent_arr = self.transformer(latent_arr)

        repres = self.avg_pooling(latent_arr)

        logits = self.classif_head(repres)

        return logits

# Postional Encodings

## Fourier Features (After failed uncommited attemps will use patch encoding for time being)

class PatchCreater(layers.Layer):
    def __init__(self, patch_size):
        super(PatchCreater, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dimens = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dimens])
        
        return patches


class PatchPosEncoder(layers.Layer):
    def __init__(self, patch_num, output_dim):
        super(PatchPosEncoder, self).__init__()
        self.patch_num = patch_num
        self.projection = layers.Dense(units=output_dim)
        self.pos_embed = layers.Embedding(
            input_dim=patch_num, output_dim=output_dim
        )

    def call(self, patches):
        posits = tf.range(start=0, limit=self.patch_num, delta=1)
        encode = self.projection(patches) + self.pos_embed(posits)
        
        return encode


