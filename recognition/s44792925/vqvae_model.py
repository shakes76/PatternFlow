
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, InputLayer, Flatten, Dense, Reshape, BatchNormalization, Dropout


class VQVAE(tf.keras.Model):

    def __init__(self, latent_dimension, kernel_size=3, strides=2):
        super(VQVAE, self).__init__()

        self.encoder = self.build_encoder(64, kernel_size, strides)
        self.decoder = self.build_decoder(64, kernel_size, strides)
        self.embedding = self.build_embedding(512, 64)

    def build_encoder(self, latent_dimension, kernel_size, strides):
        """Encoder is based off my Demo 2 VAE submission with some minor changes"""

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(256,256,1)))

        model.add(tf.keras.layers.Conv2D(64, kernel_size, strides, 'same', activation='relu')) # 'same' padding makes sure entire image is convoluted
        model.add(tf.keras.layers.BatchNormalization()) #recentering 
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2D(64, kernel_size, strides, 'same', activation='relu')) # 'same' padding makes sure entire image is convoluted
        model.add(tf.keras.layers.BatchNormalization()) #recentering 
        model.add(tf.keras.layers.LeakyReLU())

        # compress image
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(latent_dimension * 2)) 

        print(model.summary)
        return model

    def build_decoder(self, input_dimension, kernel_size, strides):
        """Decoder is based off my Demo 2 VAE submission with some minor changes"""

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(input_dimension,)))
        model.add(tf.keras.layers.Dense(8*8*64))
        model.add(tf.keras.layers.Reshape((8,8,64)))

        model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size, strides, 'same', activation='relu'))
        model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size, strides, 'same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization()) #recentering 
        model.add(tf.keras.layers.LeakyReLU()) # to avoid dead weights 

        model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size, strides, 'same', activation='relu'))
        model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size, strides, 'same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization()) #recentering
        model.add(tf.keras.layers.LeakyReLU()) # to avoid dead weights 

        model.add(tf.keras.layers.Conv2DTranspose(3, kernel_size, strides, 'same', activaiton='sigmoid'))

        print(model.summary)
        return model

    def build_embedding(self, embedding_input_dim, embedding_output_dim):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(embedding_input_dim, embedding_output_dim))

        return model


    # This function is based off the function provided at https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
    def vector_quantize(self, z):
    
        """
        z: continuous input (output of encoder)

        Returns z_quantized (discrete variable)

        shape of z: (batch_size, img_height, img_width, 1)

        We want to map z to a discrete 1 hot encoded vector which indexse
        """

        #discrete instead of continuous
        #1 hot encoding which indexes closest e
        pass


        # flatten
        z_flat = tf.reshape(z, [-1, self.embedding_dim]) # embedding dim

        # for each vector in flatten
        #    calc euclidean distance from each of the k vectors of embedding
        #    this gives us a matrix of shape (batch_size*h*w, k) 

        euc_dist = tf.math.reduce_sum(z_flat ** 2, 1, keepdims=True) + tf.math.reduce_sum(self.embedding.layers[0].weights ** 2, 1, keepdims=True) \
            - 2 * tf.linalg.matmul(z_flat, self.embedding.layers[0].weights) #possibly need to use tf.convert_to_tensor() on embedding

        # find index of closest encoding of the k vectors using argmin
        embedding_idx = tf.math.argmin(euc_dist, 1)

        # index the closest vector from the embedding for each vector
        min_embeddings = tf.math.one_hot(embedding_idx, self.num_embeddings)

        # get quantized vector
        embedding_idx = tf.reshape(embedding_idx, tf.shape(z)[:-1]) #this may be because batchsize is at the end
        z_q = self.quantize(embedding_idx)  #tf.linalg.matmul(min_embeddings, self.embedding.layers[0].weights)

        # convert back to original shape
        # dont think we need to do this with the way tf reshapes

        # copy gradients from z_q back to z_e
        z_q = z + tf.stop_gradient(z_q - z)

        # calc loss (assume beta = 0.25)
        embedding_loss = tf.reduce_mean((tf.stop_gradient(z_q)-z) ** 2) + 0.25 * tf.reduce_mean((z - tf.stop_gradient(z)) ** 2)

        return embedding_loss, z_q #, min_encodings

    # Function taken from https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
    def quantize(self, encoding_indices):
        with tf.control_dependencies([encoding_indices]):
            w = tf.transpose(self.embeddings.read_value(), [1, 0])
        return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)

    def forward(self, x):
        """Method modified versin from https://github.com/MishaLaskin/vqvae/blob/master/models/vqvae.py"""
        z_e = self.encoder(x)
        embedding_loss, z_q, perplexity = self.vector_quant(z_e)
        x_hat = self.decoder(z_q)

        return embedding_loss, x_hat, perplexity
