"""
“modules.py" containing the source code of the components of your model. Each component must be
implementated as a class or a function

Based on Neural Discrete Representation Learning by van der Oord et al https://arxiv.org/pdf/1711.00937.pdf 
and the given example on https://keras.io/examples/generative/vq_vae/
"""
import tensorflow as tf

"""CREATE STRUCTURE OF VQ-VAR MODEL"""

"""
Class Representation of the Vector Quantization laye

Structure is: 
    1. Reshape into (n,h,w,d)
    2. Calculate L2-normalized distance between the inputs and the embeddings. -> (n*h*w, d)
    3. Argmin -> find minimum distance between indices for each n*w*h vector
    4. Index from dictionary: index the closest vector from the dictionary for each of n*h*w vectors
    5. Reshape into original shape (n, h, w, d)
    6. Copy gradients from q -> x
"""
class VQ_layer(tf.keras.layers.Layer):
    def __init__(self, embedding_num, latent_dimension, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_num = embedding_num
        self.latent_dimension = latent_dimension
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(initial_value=w_init(shape=(self.latent_dimension, self.embedding_num), dtype="float32"),trainable=True,name="embeddings_vqvae",)
    
    # Forward Pass behaviour. Takes Tensor as input 
    def call(self, x):
        # Calculate the input shape and store for later -> Shape of (n,h,w,d)
        input_shape = tf.shape(x)

        # Flatten the inputs to keep the embedding dimension intact. 
        # Combine all dimensions into last one 'd' -> (n*h*w, d) 
        flatten = tf.reshape(x, [-1, self.latent_dimension])

        # Get code indices
        # Calculate L2-normalized distance between the inputs and the embeddings.
        # For each n*h*w vectors, we calculate the distance from each of k vectors of embedding dictionaty to obtain matrix of shape (n*h*w, k)
        similarity = tf.matmul(flatten, self.embeddings)
        distances = (tf.reduce_sum(flatten ** 2, axis=1, keepdims=True) + tf.reduce_sum(self.embeddings ** 2, axis=0) - 2 * similarity)

        # For each n*h*w vectors, find the indices of closest k vector from dictionary; find minimum distance.
        encoded_indices = tf.argmin(distances, axis=1)
        
        # Turn the indices into a one hot encoded vectors; index the closest vector from the dictionary for each n*h*w vector
        encodings = tf.one_hot(encoded_indices, self.embedding_num)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to its original input shape -> (n,h,w,d)
        quantized = tf.reshape(quantized, input_shape)
        
        """ LOSS CALCULATIONS """
        """
        COMMITMENT LOSS 
            Since volume of embedding spcae is dimensionless, it may grow arbitarily if embedding ei does not
            train as fast as encoder parameters. Thus add a commitment loss to make sure encoder commits to an embedding
        CODE BOOK LOSS 
            Gradients bypass embedding, so we use a dictionary learningn algorithm which uses l2 error to 
            move embedding vectors ei towards encoder output

            tf.stop_gradient -> no gradient flows through
        """
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)
        # Straight-through estimator.
        # Unable to back propragate as gradient wont flow through argmin. Hence copy gradient from qunatised to x
        # During backpropagation, (quantized -x) wont be included in computation anf the gradient obtained will be copied for inputs
        quantized = x + tf.stop_gradient(quantized - x)
        
        return quantized

"""
Returns layered model for encoder architecture built from convolutional layers. 

activations: ReLU advised as other activations are not optimal for encoder/decoder quantization architecture.
e.g. Leaky ReLU activated models are difficult to train -> cause sporadic loss spikes that model struggles to recover from
"""
# Encoder Component
def encoder_component(latent_dimension):
    #2D Convolutional Layers
    # filters -> dimesion of output space
    # kernal_size -> convolution window size
    # activation -> activation func used
        # relu ->
    # strides -> spaces convolution window moves vertically and horizontally 
    # padding -> "same" pads with zeros to maintain output size same as input size
    inputs = tf.keras.Input(shape=(256, 256, 1))

    layer = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    layer = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(layer)
    
    outputs = tf.keras.layers.Conv2D(latent_dimension, 1, padding="same")(layer)
    return tf.keras.Model(inputs, outputs, name="encoder")

"""
Returns the model for decoder architecture built from  tranposed convolutional layers. 

activations: ReLU advised as other activations are not optimal for encoder/decoder quantization architecture.
e.g. Leaky ReLU activated models are difficult to train -> cause sporadic loss spikes that model struggles to recover from
"""
# Decoder Component
def decoder_component(latent_dimension):
    inputs = tf.keras.Input(shape=encoder_component(latent_dimension).output.shape[1:])
    layer = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(inputs)
    layer = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(layer)
    outputs = tf.keras.layers.Conv2DTranspose(1, 3, padding="same")(layer)
    return tf.keras.Model(inputs, outputs, name="decoder")

# Build Model
def build_model(embeddings_num, latent_dimension):
    vq_layer = VQ_layer(embeddings_num, latent_dimension, name="vector_quantizer")
    encoder = encoder_component(latent_dimension)
    decoder = decoder_component(latent_dimension)
    inputs = tf.keras.Input(shape=(256, 256, 1))
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return tf.keras.Model(inputs, reconstructions, name="vq_vae")

# Create a model instance and sets training paramters 
class vqvae_model(tf.keras.models.Model):
    def __init__(self, variance, latent_dimension, embeddings_num, **kwargs):
        
        super(vqvae_model, self).__init__(**kwargs)
        self.latent_dimension = latent_dimension
        self.embeddings_num = embeddings_num
        self.variance = variance
        
        self.model = build_model(embeddings_num, latent_dimension)

        self.total_loss = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.vq_loss = tf.keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        # Model metrics -> returns losses (total loss, reconstruction loss and the vq_loss)
        return [self.total_loss, self.reconstruction_loss, self.vq_loss]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.model(x)

            # Calculate the losses.
            reconstruction_loss = (tf.reduce_mean((x - reconstructions) ** 2) / self.variance)
            total_loss = reconstruction_loss + sum(self.model.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Loss tracking.
        self.total_loss.update_state(total_loss)
        self.reconstruction_loss.update_state(reconstruction_loss)
        self.vq_loss.update_state(sum(self.model.losses))

        # Log results.
        return {
            "loss": self.total_loss.result(),
            "reconstruction_loss": self.reconstruction_loss.result(),
            "vqvae_loss": self.vq_loss.result(),
        }