import tensorflow as tf
import math
from tensorflow.keras import layers
from tensorflow import keras

'''
Defines Perceiver model and fourier feature code.
'''

# GET FOURIER FEATURES FOR POSITIONAL ENCODINGS 

# img_data: tensor of shape (datapoints, rows, cols, 1)
def get_positional_encodings(img_data, bands=4, sampling_rate=10):
    ''' (Doesn't work, slightly off.)
    # assume 2 dimensions, using single channel images
    data_points, rows, cols = img_data.shape
    xr, xc = tf.linspace(-1,1,rows), tf.linspace(-1,1,cols)
    xd = tf.expand_dims(tf.reverse(tf.meshgrid(xr,xc), axis=[-3]),3)
    xd = tf.reshape(tf.concat([xd[0], xd[1]], axis=2),(rows,cols,2))
    xd = tf.repeat(tf.expand_dims(xd, -1), repeats=[2*bands + 1], axis=3) # (rows, cols, 2, 2F + 1)
    # logscale for frequencies ( * pi) , 0 start as 10**0 = 1
    frequencies = tf.experimental.numpy.logspace(0.0,(tf.math.log(sampling_rate/2)/tf.math.log(10.)), num = bands, dtype = tf.float32) * math.pi
    # (228,260,2,9)
    f_features = tf.cast(xd, tf.float32)
    f_features = tf.concat([tf.math.sin(f_features[:,:,:,0:4] * frequencies), tf.math.cos(f_features[:,:,:,4:8] * frequencies), tf.expand_dims(f_features[:,:,:,8], -1)], axis=-1)
    f_features = tf.repeat(tf.reshape(f_features, (1,rows,cols,2*(2*bands + 1))), repeats=[data_points],axis=0) # (data_points, 228, 260, 18)
    f_features = tf.cast(f_features, tf.float32)
    return tf.reshape(tf.concat((tf.expand_dims(tf.cast(img_data, tf.float32), 3),f_features),axis=-1), (data_points, rows*cols, -1)) # add data in and flatten images
    '''
    # Fourier Encoding partially from https://github.com/Rishit-dagli/Perceiver, refer to comments above
    b, *axis, _ = img_data.shape
    axis_pos = list(map(lambda size: tf.linspace(-1.0, 1.0, num=size), axis))
    pos = tf.stack(tf.meshgrid(*axis_pos, indexing="ij"), axis=-1)
    x = tf.expand_dims(pos, -1)
    x = tf.cast(x, dtype=tf.float32)
    orig_x = x
    scales = tf.experimental.numpy.logspace(
        0.0,
        math.log(sampling_rate / 2) / math.log(10.),
        num=bands,
        dtype=tf.float32
    )
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    x = x * scales * math.pi
    x = tf.concat([tf.math.sin(x), tf.math.cos(x)], axis=-1)
    x = tf.concat((x, orig_x), axis=-1)
    encoding = tf.repeat(tf.reshape(x, (1, 228, 260, 2 * (2 * bands + 1))), repeats = b, axis=0)
    return tf.reshape(tf.concat((img_data, encoding), axis=-1), (b, 228*260, -1))

''' Returns the cross attention QKV model. '''
def get_attention_module(channel_size, data_size, latent_size):

    inputs = [layers.Input((latent_size, channel_size)),layers.Input((data_size, channel_size))]

    # Q, K & V linear networks
    query_mlp = inputs[0]
    query_mlp = layers.LayerNormalization()(query_mlp)
    latent_output = query_mlp
    query_mlp = layers.Dense(channel_size)(query_mlp)

    key_mlp = inputs[1]
    key_mlp = layers.LayerNormalization()(key_mlp)
    key_mlp = layers.Dense(channel_size)(key_mlp)

    value_mlp = inputs[1]
    value_mlp = layers.LayerNormalization()(value_mlp)
    value_mlp = layers.Dense(channel_size)(value_mlp)

    # QKV cross-attention
    attention_module = layers.Attention(use_scale=True)([query_mlp, key_mlp, value_mlp])
    attention_module = layers.Dense(channel_size)(attention_module)
    attention_module = layers.Add()([latent_output, attention_module])
    attention_module = layers.LayerNormalization()(attention_module)

    # New query from attention module 
    new_latent = layers.Dense(channel_size, activation=tf.nn.gelu)(attention_module)
    #new_latent = layers.Dense(channel_size, activation=tf.nn.gelu)(new_latent)
    new_latent = layers.Dense(channel_size)(new_latent)
    new_latent = layers.Add()([attention_module, new_latent])

    cross_attention = keras.Model(inputs=inputs, outputs = new_latent)
    return cross_attention

''' Returns transformer model, using multihead attention. '''
def get_transformer_module(latent_size, channel_size, transformer_heads):
    latent_input = layers.Input((latent_size, channel_size))
    layer_init = latent_input
    for i in range(4): # 4 transformer blocks
        transformer = layers.LayerNormalization()(layer_init)
        transformer = layers.MultiHeadAttention(num_heads = transformer_heads, key_dim = channel_size)(transformer, transformer, \
            return_attention_scores = False)
        transformer = layers.Add()([latent_input, transformer])
        transformer = layers.LayerNormalization()(transformer)
        
        new_query = layers.Dense(channel_size, activation=tf.nn.gelu)(transformer)
        #new_query = layers.Dense(channel_size, activation=tf.nn.gelu)(new_query)
        new_query = layers.Dense(channel_size)(new_query)
        transformer = layers.Add()([new_query, transformer])
        layer_init = transformer

    return keras.Model(inputs = latent_input, outputs = transformer)

''' Truncated intializer for initial latent array. For details refer to Perceiver paper. '''
def truncated_initializer(shape, dtype=None):
    norm = tf.random.normal(shape, mean=0.0, stddev=0.02, dtype=dtype)
    # truncation
    return tf.math.minimum(tf.math.maximum(norm, tf.constant(-2, dtype=tf.float32, shape=norm.shape)),tf.constant(2, dtype=tf.float32, shape=norm.shape)) 

''' Ensures data is divisible/within range of batch size, if not removes data off the end. '''
def create_batches_from_data(xdata, ydata, batches):
    xdata_new = xdata[:int(len(xdata) / batches) * batches]
    ydata_new = ydata[:int(len(ydata) / batches) * batches]
    return (xdata_new, ydata_new)

''' Perceiver model class. '''
class Perceiver(tf.keras.Model):
    def __init__(self, latent_size = 64, data_size = 228*260, bands = 4, transformer_heads = 4, 
                sampling_rate = 10, iterations = 4):
        super(Perceiver, self).__init__()
        self.bands = bands
        self.latent_size = latent_size
        self.data_size = data_size
        self.transformer_heads = transformer_heads
        self.channel_size = 2*(2*bands + 1) + 1 # data (1) + 2 dim * (2F + 1)
        self.sampling_rate = sampling_rate
        self.iterations = iterations
        self.init_latent = self.add_weight(shape=(self.latent_size, self.channel_size), initializer= truncated_initializer, trainable=True)
        self.init_latent = tf.reshape(self.init_latent, (1,*self.init_latent.shape))
        self.attention_module = get_attention_module(channel_size=self.channel_size, data_size=self.data_size, latent_size=self.latent_size)
        self.transformer_module = get_transformer_module(self.latent_size, self.channel_size, self.transformer_heads)
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classify = layers.Dense(1, activation='sigmoid')# binary crossentropy

    def call(self, xdata):
        # get fourier features and add them onto the data
        encoded_data = get_positional_encodings(xdata, bands=self.bands, sampling_rate=self.sampling_rate)
        input_data = [self.init_latent, encoded_data]
        # add each iteration of cross attention and transformers
        for layer_num in range(self.iterations):
            new_latent = self.attention_module(input_data)
            new_query = self.transformer_module(new_latent)
            # tranformer module outputs new query for next attention module
            input_data[0] = new_query
        # return classification (left/right) after average pooling
        return self.classify(self.global_pool(new_query))
    
    ''' Trains perceiver from data. '''
    def train_model(self, xtrain, ytrain, xval, yval, epochs, batches, lr_func): 
        X_train, y_train = create_batches_from_data(xtrain, ytrain, batches)
        X_val, y_val = create_batches_from_data(xval, yval, batches)

        history = self.fit(
			X_train, y_train,
			epochs = epochs,
			batch_size = batches,
            callbacks=[lr_func],
            validation_data = (X_val, y_val),
            validation_batch_size = batches
		)
        return history
