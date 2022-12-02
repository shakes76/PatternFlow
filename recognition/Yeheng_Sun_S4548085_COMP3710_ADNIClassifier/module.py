# import packages
import tensorflow as tf
from tensorflow import keras
from keras.layers import Normalization, Resizing, RandomFlip, RandomRotation, RandomZoom, Dense, Dropout, \
    Layer, Embedding, Input, LayerNormalization, MultiHeadAttention, Add, LayerNormalization, Flatten

img_size = 256  # must match train.py and dataset.py image size



# define MLP given a list which record the number of nodes in each layer
def multi_layer_preceptron(x, layer_list, drop_out_rate):
    for layer_node in layer_list:
        x = Dense(layer_node, activation=tf.nn.gelu)(x)
        x = Dropout(drop_out_rate)(x)
    return x


# a class used to encode each patch into vector
class Patch2Vec(Layer):
    def __init__(self, patch_n, proj_vec_n):
        super(Patch2Vec, self).__init__()
        self.patch_n = patch_n
        self.proj_layer = Dense(units=proj_vec_n)
        self.position_embed_layer = Embedding(
            input_dim=patch_n, output_dim=proj_vec_n
        )

    def call(self, patch):
        position = tf.range(start=0, limit=self.patch_n, delta=1)
        encode_vec = self.proj_layer(patch) + self.position_embed_layer(position)
        return encode_vec


# a class to split images into patches
class Patches(Layer):
    def __init__(self, patch_len):
        super(Patches, self).__init__()
        self.patch_len = patch_len

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_len, self.patch_len, 1],
            strides=[1, self.patch_len, self.patch_len, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_n = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_n])
        return patches


def vision_transformer(input_shape, patch_len, patch_n, proj_vec_n, transformer_n, head_n, class_n,
                       transformer_units, mlp_head_units):

    # data augmentation and patch operation
    input_img = Input(shape=input_shape)
    patch_img = Patches(patch_len)(input_img)
    patch_vec = Patch2Vec(patch_n, proj_vec_n)(patch_img)

    # transformer modules
    for _ in range(transformer_n):
        x1 = LayerNormalization()(patch_vec)
        attention_output = MultiHeadAttention(
            num_heads=head_n, key_dim=proj_vec_n, dropout=0.1
        )(x1, x1)
        x2 = Add()([attention_output, patch_vec])
        x3 = LayerNormalization()(x2)
        x3 = multi_layer_preceptron(x3, layer_list=transformer_units, drop_out_rate=0.1)
        patch_vec = Add()([x3, x2])

    # MLP classifier
    feature = LayerNormalization()(patch_vec)
    feature = Flatten()(feature)
    feature = Dropout(0.5)(feature)
    feature = multi_layer_preceptron(feature, layer_list=mlp_head_units, drop_out_rate=0.5)
    output = Dense(class_n, activation='sigmoid')(feature)
    model = keras.Model(inputs=input_img, outputs=output)
    return model
