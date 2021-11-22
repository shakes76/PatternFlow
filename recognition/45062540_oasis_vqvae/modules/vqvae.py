import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class VectorQuantizer(tf.keras.layers.Layer):
    """
    Create the Vector quantizer layer (custom layer).
    """

    def __init__(self, num_embeddings, embedding_dim, beta: float = 0.25, **kwargs):
        """
        Create a vector quantizer layer
        
        Params:
            num_embeddings(int): number of embeddings in the codebook (discrete latent space)
            embedding_dim(int): the dimensionality of each latent embedding vector
            beta(int): used when calculating the loss, best kept between 0.1 to 2, default to 0.25
            **kwargs: additional keyword arguments
        """
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        #Initialize the embeddings that we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value = w_init(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable = True, name = "dictionary")

    def call(self, inputs):
        """
        Customize the forward pass behavior
        
        Params:
            inputs(tf.Tensor): the input data
            
        Returns:
            (tf.Tensor): the embedding closet to the inputs in the codebook
        """
        # Get the input shape of the inputs
        input_shape = tf.shape(inputs)
        # Flatten the inputs while keeping embedding_dim
        flattened = tf.reshape(inputs, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add it to the layer.
        commitment_loss = self.beta * tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        # The quantization process is not differentiable. Create a straight-through estimator between the decoder 
        # and the encoder s.t. the decoder gradients are directly propagated to the encoder.
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        return quantized

    def get_code_indices(self, flattened_inputs):
        """
        Calculate the L2-normalized distance between the inputs and the embeddings in the codebook
        
        Params:
            flattened_inputs(tf.Tensor): the input data flattened
        
        Returns:
            (tf.Tensor): the encoding indices with the field (index of embedding) has the minimum distance to the input set to 1, 
            all other fields equals to 0
        """
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

class Encoder(keras.models.Model):
    """
    Create the VQ-VAE Encoder
    """
    def __init__(self, latent_dim = 256, **kwargs):
        """
        Create a VQ-VAE encoder
        
        Params:
            latent_dim(int): the dimensionality of the ouput of the encoder
            **kwargs: additional keyword arguments
        """
        super().__init__(**kwargs)
        self.conv_layers = [
            layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"), 
            layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"), 
            layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")]
        self.conv_final = layers.Conv2D(filters=latent_dim,kernel_size=1,padding="same")

    def call(self, inputs):
        """
        Customize the forward pass behavior
        
        Params:
            inputs(tf.Tensor): the input data
            
        Returns:
            (tf.Tensor): the output of the encoder
        """
        out = inputs
        for layer in self.conv_layers:
            out = layer(out)
        out = self.conv_final(out)
        return out

class Decoder(keras.models.Model):
    """
    Create the VQ-VAE Decoder
    """
    def __init__(self, **kwargs):
        """
        Create a VQ-VAE decoder
        
        Params:
            **kwargs: additional keyword arguments
        """
        super().__init__(**kwargs)
        self.conv_layers = [
            layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same"), 
            layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        ]
        self.conv_final = layers.Conv2DTranspose(1, 3, padding="same")

    def call(self, inputs):
        """
        Customize the forward pass behavior
        
        Params:
            inputs(tf.Tensor): the input data
            
        Returns:
            (tf.Tensor): the output of the decoder
        """
        out = inputs
        for layer in self.conv_layers:
            out = layer(out)
        out = self.conv_final(out)
        return out

class VQVAE(tf.keras.Model):
    """
    Create the VQ-VAE model
    """
    def __init__(self,num_embeddings = 256,latent_dim = 256,**kwargs):
        """
        Create a VQ-VAE model
        
        Params:
            num_embeddings(int): number of embeddings in the codebook (discrete latent space)
            embedding_dim(int): the dimensionality of each latent embedding vector
            **kwargs: additional keyword arguments
        """
        super().__init__(**kwargs)
        self.vq_layer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dim, name="vector_quantizer")
        self.encoder = Encoder(latent_dim=latent_dim, name="encoder")
        self.decoder = Decoder(name="decoder")

        self.total_loss = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.vq_loss = tf.keras.metrics.Mean(name="vq_loss")

    def call(self, inputs, training=False):
        """
        Customize the forward pass behavior.
        
        Params:
            inputs(tf.Tensor): the input data
            training(Boolean): indicate whether the layer should behave in training mode or in inference mode.
                    training = False, use the moving mean and the moving variance to normalize the current batch, 
                    rather than using the mean and variance of the current batch.
        
        Returns:
            (tf.Tensor): the output of the decoder
        """
        encoder_outputs = self.encoder(inputs, training=training)
        quantized_latents = self.vq_layer(encoder_outputs, training=training)
        reconstructions = self.decoder(quantized_latents, training=training)
        return reconstructions

    @property
    def metrics(self):
        """
        Model metrics
        
        Returns:
            the losses (total loss, reconstruction loss and the vq_loss)
        """
        return [self.total_loss, self.reconstruction_loss, self.vq_loss]

    def train_step(self, inputs):
        """
        Customize the the logic of a training step(calculate losses, backpropagation, and update metrics)
        
        Params:
            inputs(tf.Tensor): the input data
            
        Returns:
            (tf.Tensor): the metric values
        """
        with tf.GradientTape() as tape:
            # Output from the VQ-VAE.
            reconstructions = self(inputs, training=True)

            # Calculate the losses.
            reconstruction_loss = tf.reduce_mean((inputs - reconstructions) ** 2)
            total_loss = reconstruction_loss + sum(self.vq_layer.losses)

        # Compute the gradients
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        
        # Update the weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Update the metrics
        self.total_loss.update_state(total_loss)
        self.reconstruction_loss.update_state(reconstruction_loss)
        self.vq_loss.update_state(sum(self.vq_layer.losses))

        # Log results.
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, inputs):
        """
        Customize the the logic of a testing step (calculate losses)
        
        Params:
            inputs(tf.Tensor): the input data
            
        Returns:
            (tf.Tensor): the metric values
        """
        reconstructions = self(inputs, training=True)

        # Calculate the losses.
        reconstruction_loss = tf.reduce_mean((inputs - reconstructions) ** 2)
        total_loss = reconstruction_loss + sum(self.vq_layer.losses)

        # Update metrics
        self.total_loss.update_state(total_loss)
        self.reconstruction_loss.update_state(reconstruction_loss)
        self.vq_loss.update_state(sum(self.vq_layer.losses))

        # Log results.
        return {metric.name: metric.result() for metric in self.metrics}