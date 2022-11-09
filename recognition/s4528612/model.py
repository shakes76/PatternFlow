import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array

"""
Produces Attention Component of Cross-Attention Layer
Returns: Attention Component of Cross-Attention Layer
"""
def attention_component(inputs,input_normalized):
    #Produces the attention layers
    attention_layer = layers.MultiHeadAttention(8, 27)(input_normalized, input_normalized)

    attention_layer = layers.Dense(27)(attention_layer)

    attention_layer = layers.Add()([attention_layer, inputs])

    attention_layer = layers.LayerNormalization()(attention_layer)

    return attention_layer, layers.Dense(27, activation=tf.nn.gelu)(attention_layer)

"""
Returns the cross-attention layer
Returns: Cross-attention Layer
"""
def cross_attention():
    image_size = 73 * 64
    # Number of Pixels in the Scaled Image
    print(image_size)
    latent_input = layers.Input(shape=(256, 2*(2*6 + 1) + 1))
    data_array_input = layers.Input(shape=(image_size, 2*(2*6 + 1) + 1))

    latent_array = layers.LayerNormalization(epsilon=1e-6)(latent_input)
    data_array = layers.LayerNormalization(epsilon=1e-6)(data_array_input)

    # Produce Querys Keys and Values of inputs
    query_key_value_vector = []
    

    query_key_value_vector.append(layers.Dense(units=2*(2*6 + 1) + 1)(latent_array))
    for _ in range(2):
        query_key_value_vector.append(layers.Dense(units=2*(2*6 + 1) + 1)(data_array))

    attention = layers.Attention(use_scale=True, dropout=0.1)(
        query_key_value_vector
    )
    # Add Input to Output
    attention = layers.Add()([attention, latent_array])
    attention = layers.LayerNormalization(epsilon=1e-6)(attention)
    
    # Feedforward network.
    feedforward_network = [] # May need to add more layers
    feedforward_network.append(layers.Dense(units=2*(2*6 + 1) + 1))
    outputs = keras.Sequential(feedforward_network)(attention)
    
    outputs = layers.Add()([outputs, attention])

    return keras.Model(inputs=[latent_input,data_array_input], outputs=outputs)

"""
Given an image, a Fourier Encoding is applied to the image to retain position data.
Params:
    image: Input image to have the Fourier Encoding applied.
Returns: Image with Fourier Encoding applied
"""
def fourier_encode(image):
    batch_size, *axis, _ = image.shape
    axis_pos = list(map(lambda size: tf.linspace(-1.0, 1.0, num=size), axis))
    array = tf.stack(tf.meshgrid(*axis_pos, indexing="ij"), axis=-1)   
    array = tf.cast(tf.expand_dims(array,-1), dtype=tf.float32)
    array_copy = array # Tensors aren't reference types 
    #Fourier Encoding Calculation
    encode = tf.reshape(tf.experimental.numpy.logspace(start=0.0,stop=0.69, num=6, dtype=tf.float32,),(*((1,) * (len(array.shape) - 1)), 6)) 
    array =  3.14 * array * encode
    # Repeat the Fourier Encoding For the number of times in the batch and reshape to the correct dimension.
    layer = tf.repeat(tf.reshape(tf.concat((tf.concat([tf.math.sin(array), tf.math.cos(array)], axis=-1),array_copy),axis=-1),
                       (1, image.shape[1], image.shape[2], 2*(2*6+1))), repeats=image.shape[0], axis=0)

    return tf.reshape(tf.concat((image, layer), axis=-1), (image.shape[0], image.shape[1]*image.shape[2], -1)) 

"""
Creates transformer layer
Returns: Transformer Layer
""" 
def transformer_layer():
    # Initialise component
    inputs = layers.Input(shape=(256, 27))
    input_normalized = layers.LayerNormalization()(inputs)
    i = 0
    while i < 6:
        input_output_sum = layers.Add()([layers.Dense(27)(attention_component(inputs,input_normalized)[1]), attention_component(inputs,input_normalized)[0]])
        print(i)
        i += 1
    return tf.keras.Model(inputs=inputs, outputs=input_output_sum)

"""
Overall Perceiver Model Class
"""
class Perceiver(tf.keras.Model):

    def __init__(self):
        super(Perceiver, self).__init__()

    """
        Builds the Perceiver
        Params:
            Input_shape: Input Shape of the Image
    """
    def build(self, input_shape):
        self.latent_array = self.add_weight(
            shape=(256, 27),
            initializer="random_normal",
            trainable=True,
            name='latent'
        )
        # Define Required Layers in Model
        
        # Cross Attention Layer
        self.cross_attention = cross_attention()
        # Transformer Layer
        self.transformer = transformer_layer()
        # Global Pooling Layer
        self.global_average_pooling = layers.GlobalAveragePooling1D()
        # Classifier Output
        self.classify = layers.Dense(units=1, activation=tf.nn.sigmoid)
        super(Perceiver, self).build(input_shape)

    def call(self, inputs):
        cross_attention = [
            tf.expand_dims(self.latent_array, 0),
            fourier_encode(inputs)
        ]
        i = 0
        # Produces 8 Transformer layers
        while i < 8:
            latent_array = self.transformer(self.cross_attention(cross_attention))
            cross_attention[0] = latent_array
            i += 1
        return self.classify(self.global_average_pooling(cross_attention[0]))