#VQ-VAE parameters from the paper
beta = 0.25 #commitment loss weighting - based on the paper
K = 512 #Number of codebook vectors / embeddings number - using recommended value in the paper

# Vector Quantizer Layer
class Vector_Quantizer(tf.keras.layers.Layer):

    def call(self, encoder_outputs):
        #define the size of our embeddings - has same length as latent space vectors, and width as defined by K (hyperparameter)
        codebook_shape = (latent_dims, K)
        #create the embedding object - initialise uniformly
        initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
        #Declare our embeddings as a tensorflow variable that will be learnt
        embeddings = tf.Variable(initializer(shape=codebook_shape, dtype='float32'), trainable = True)

        #Flatten all dimensions of the encoded vectors except the channels
        #Encoded vectors = (B,H,W,C) -> (B*H*W, C) i.e. flattened, each of which will be quantized independently
        encoded_vectors = tf.reshape(encoder_outputs, [-1, codebook_dims])

        #Now can quantize each
        quantized_vectors = quantize_vectors(encoded_vectors)
        
        # Reshape the flat vectors back into 2D for use in the convolutional decoder network
        reshaped_quantized_vectors = tf.reshape(quantized_vectors, tf.shape(encoder_outputs))
        return reshaped_quantized_vectors

    def quantize_vectors(self, vectors, embeddings):
        #First need to calculate the distance between our input vectors and every vector in the codebook
        # Paper uses L2 norm: sqrt((x2-x1)^2 + (y2-y1)^2 +...)
        distances = tf.reduce_sum(vector**2, axis=1, keepdims=True) - 2*tf.matmul(vector, embeddings) + tf.reduce_sum(embeddings**2, axis=0, keepdims=True) 

        # Can now determine to which codebook vector our input vectors are closest to (minimum distance)
        indices = tf.argmin(distances, 1)
        
        #Do one-hot encoding so that only the closest codebook vector is mapped to a 1, all others to a 0
        one_hot_indices = tf.one_hot(indices, K)
        
        # Apply indices to the embeddings - now have quantized the vectors
        quantized_vectors = tf.mamul(one_hot_indices, embeddings, transpose_b=True)
        
        return quantized_vectors

# Define the encoder and decoder networks
def encoder():

    # Encoder network - takes input images and encodes them into z
    inputs = tf.keras.layers.Input(shape=(image_height, image_width, channels))
    
    # First downsample by half
    conv = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation='relu', padding = "same")(inputs)
    #Second downsample by half
    conv = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, activation='relu', padding = "same")(conv)
    #Third downsample by half
    #conv = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, activation='relu', padding = "same")(conv)
    
    #Final conv layer - image is now encoded into size of latent space
    hidden = tf.keras.layers.Conv2D(latent_dims, kernel_size = 1, padding = "same")(conv)
    
    #Quantize the vectors through the Vector Quantizer Layer
    output = Vector_Quantizer()(hidden)

    # Build the encoder
    final_network = tf.keras.Model(inputs, output)

    return final_network
    
    
#def encoder_loss():


def decoder():
    # Takes the latent space and upsamples until original size is reached
    #256 -> downsampled twice = 256/4 = 64
    inputs = tf.keras.layers.Input(shape=(latent_dims,))
    
    # Now can upsample twice
    conv = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(dense)
    conv = tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation='relu')(conv)
    
    #Output layer
    output = tf.keras.layers.Conv2D(1,kernel_size=[1,1], strides=1, activation='relu', padding = "same")(conv)
    
    #build the decoder
    final_network = tf.keras.Model(inputs, output)
    return final_network

#def decoder_loss()

def VQ_VAE_network():
    #Create the encoder and decoder components of VQ-VAE Model
    encoder_network = encoder()
    decoder_network = decoder()

    # Construct the overall VQ-VAE now
    inputs = tf.keras.layers.Input((image_height, image_width, channels))
    hidden = encoder_network(inputs)
    output = decoder_network(hidden)

    vq_vae = tf.keras.Model(inputs=inputs, outputs=output)
    optimizer = tf.keras.optimizers.Adam(1e-4)