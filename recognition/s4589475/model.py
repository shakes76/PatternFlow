"""
Model script containing all VQ-VAE model and pixelCNN model related functions.
This includes the functions for building the model, training the model, and displaying the models' results.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# OASIS Dataset constants
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 1 # images are in grayscale

# Pixel CNN Constants
PIXELCNN_INPUT_SHAPE = (32,32)

# Vector Quantizer Layer for the VQVAE
class Vector_Quantizer(tf.keras.layers.Layer):
    
    def __init__(self, latent_dims, K):
        """
        Initializes a new Vector Quantization layer with uniformly distributed codebook.
        """
        super().__init__()
        #define the size of our embeddings - has same length as latent space vectors, and width as defined by K (hyperparameter)
        codebook_shape = (latent_dims, K)
        #create the embedding object - initialise uniformly
        initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
        #Declare our embeddings as a tensorflow variable that will be learnt
        self._embeddings = tf.Variable(initializer(shape=codebook_shape, dtype='float32'), trainable = True)

    def quantize_vectors(self, vectors):
        """
        Takes the VQ-VAE encoder outputs (continuous vectors) and quantizes them by "snapping" them to the nearest codebook vector.
        Returns the quantized vector and its corresponding index.
        """
        #First need to calculate the distance between our input vectors and every vector in the codebook
        # Paper uses L2 norm: sqrt((x2-x1)^2 + (y2-y1)^2 +...)
        distances = tf.reduce_sum(vectors**2, axis=1, keepdims=True) - 2*tf.matmul(vectors, self._embeddings) + tf.reduce_sum(self._embeddings**2, axis=0, keepdims=True) 

        # Can now determine to which codebook vector our input vectors are closest to (minimum distance)
        indices = tf.argmin(distances, 1)
        
        #Do one-hot encoding so that only the closest codebook vector is mapped to a 1, all others to a 0
        one_hot_indices = tf.one_hot(indices, K)
        
        # Apply indices to the embeddings - now have quantized the vectors
        quantized_vectors = tf.matmul(one_hot_indices, self._embeddings, transpose_b=True)
        
        return indices, quantized_vectors

    def call(self, encoder_outputs):
        """
        Quantizes the given VQ-VAE encoders' output vectors using the codebook.
        Returns the quantized vectors with and without straight-through estimation applied.
        """

        #Flatten all dimensions of the encoded vectors except the channels
        #Encoded vectors = (B,H,W,C) -> (B*H*W, C) i.e. flattened, each of which will be quantized independently
        encoded_vectors = tf.reshape(encoder_outputs, [-1, latent_dims])

        #Now can quantize each
        indices, quantized_vectors = self.quantize_vectors(encoded_vectors)
        
        # Reshape the flat vectors back into 2D for use in the convolutional decoder network
        reshaped_quantized_vectors = tf.reshape(quantized_vectors, tf.shape(encoder_outputs))
        
        #Do straight-through estimation so that back-propagation can occur
        # Going forward, output = encoder_output + quantized_vectors - encoder_output = quantized vectors (as desired)
        # Going backward, we copy the gradients at the decoder back to the encoder
        output = encoder_outputs + tf.stop_gradient(reshaped_quantized_vectors - encoder_outputs)
        return output, reshaped_quantized_vectors

def encoder_network(latent_dims):
    """
    Creates and returns an encoder neural network model.
    Compresses the input image via strided convolutional layers into a latent space representation (z).
    """
    inputs = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    
    # First downsample by half
    conv = tf.keras.layers.Conv2D(16, kernel_size=3, strides=2, activation='relu', padding = "same")(inputs)
    #Second downsample by half
    conv = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation='relu', padding = "same")(conv)
    #Third downsample by half
    conv = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, activation='relu', padding = "same")(conv)
    
    #Final conv layer - image is now encoded into size of latent space
    output = tf.keras.layers.Conv2D(latent_dims, kernel_size = 1, padding = "same")(conv)

    # Build the encoder
    final_network = tf.keras.Model(inputs, output, name="Encoder")
    
    print(final_network.summary())
    return final_network

def decoder_network(latent_dims):
    """
    Creates and returns a decoder neural network model. 
    Takes a quantized latent space as input and upsamples until a reconstructed image of original size is achieved.
    """
    #256 -> downsampled thrice = 256/8 = 32
    inputs = tf.keras.layers.Input(shape=(32,32,latent_dims))
    
    # Now can upsample twice
    conv = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    conv = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(conv)
    conv = tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation='relu')(conv)

    #Output layer
    output = tf.keras.layers.Conv2D(1,kernel_size=[1,1], strides=1, activation='relu', padding = "same")(conv)
    
    #build the decoder
    final_network = tf.keras.Model(inputs, output, name="Decoder")
    print(final_network.summary())

    return final_network

def reconstruction_loss(inputs, outputs): 
    """
    Returns the sum of the Mean Squared Error, as well as Structural Similarity Loss, between the input image and the output reconstruction
    """ 
    #Calculate mse loss
    mse = tf.keras.losses.MeanSquaredError()
    recon_loss = mse(inputs, outputs)
    
    #Calculate ssim to use as part reconstruction loss - focusses on the brain rather than background
    ssim_loss = 1 - tf.image.ssim(inputs, outputs, max_val = 1)
    
    return recon_loss + ssim_loss

def latent_loss(encoder_output, quantized_vectors, beta):
    """
    Returns the latent loss - replaces kl-divergence in traditional VAE with two terms (as kl-divergence can't be minimized (is a constant))
    Latent loss is the sum of the codebook and committment losses, both of which are Mean Squared Errors.
    """
    # latent commitment loss
    # beta*||z_e(x)-sg[e]|| i.e. freeze the codebook vectors/embeddings and push the encoded vectors towards codebook vectors
    mse1 = tf.keras.losses.MeanSquaredError()
    e_loss = beta * mse1(tf.stop_gradient(quantized_vectors), encoder_output)

    # latent codebook loss
    #||sg[z_e(x)]-e|| i.e. freeze the encoded vectors and push codebook vectors to encoded vectors
    mse2 = tf.keras.losses.MeanSquaredError()
    q_loss = mse2(quantized_vectors, tf.stop_gradient(encoder_output))
    
    return q_loss + e_loss

def create_overall_vqvae(encoder_net, quantizer, decoder_net):
    """
    Creates the overall vq-vae neural network model.
    Builds together an encoder network followed by Vector Quantizer layer, then the decoder network.
    """
    inputs = tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    encoder_outputs = encoder_net(inputs)
    quantized_latents, z1 = quantizer(encoder_outputs)
    reconstructions = decoder_net(quantized_latents)
    # Build the overall model and return it
    vq_vae_model_overall = tf.keras.Model(inputs, reconstructions, name="overall model")
    
    return vq_vae_model_overall


@tf.function
def training_step(images, optimizer, encoder, quantizer_layer, decoder, vq_vae_model_overall, beta):
    """
    A custom training step for the VQ-VAE.
    Uses a gradient tape to store the gradients, which are applied to the optimizer to improve the model.
    Returns the overall loss, as well the constituent reconstruction and latent losses.
    """

    with tf.GradientTape(persistent=True) as vae_tape:
        #Get the latent space
        z = encoder(images, training=True)
        #Get the quantized latent space
        z_quantized, z1 = quantizer_layer(z, training=True)
        #Get the reconstructions
        reconstructions = decoder(z_quantized, training=True)        

        #determine overall loss: recon_loss + latent_loss
        recon_loss = tf.reduce_mean(reconstruction_loss(images, reconstructions) )
        #recon_loss = reconstruction_loss(images, reconstructions)
        latent_loss1 = tf.reduce_mean(latent_loss(z, z1, beta))
        
        total_loss = recon_loss + latent_loss1    
    
    gradients = vae_tape.gradient(total_loss, vq_vae_model_overall.trainable_variables)
    
    #apply the gradients to the optimizer
    optimizer.apply_gradients(zip(gradients, vq_vae_model_overall.trainable_variables))
    
    return recon_loss, latent_loss1, total_loss


def train(encoder_net, decoder_net, quantizer_net, opt, vq_model, beta, epoch):
    """
    Trains the VQ-VAE model for the given number of epochs.
    Plots the training loss curves upon completion.
    """

    recon_loss_list = []
    latent_loss_list = []

    losses = []
    for epoch in range(1, epochs+1):
        print(epoch)
        loss = -1
        batch_losses = 0

        batch_losses_recon = 0
        batch_losses_latent = 0

        count = 0

        for image_batch in training_ds:

            recon_loss, latent_loss_results, total_loss = training_step(image_batch, opt, encoder_net, quantizer_net, decoder_net, vq_model, beta)

            batch_losses += total_loss
            batch_losses_recon += recon_loss
            batch_losses_latent += latent_loss_results
            count += 1 

            losses.append(batch_losses/count)
            recon_loss_list.append(batch_losses_recon/count)
            latent_loss_list.append(batch_losses_latent/count)
    
    
    # Plot the loss curves
    plt.title('Loss curves for training')
    plt.plot(losses)
    plt.plot(recon_loss_list)
    plt.plot(latent_loss_list)

    plt.show()

def calculate_ssims(vq_vae_model_overall, testing_ds_batched):
    """
    Calculates the average Structural Similarity Score across the entire testing dataset.
    Also displays a comparison of the original and reconstructed images with associated SSIM for every 100th image.
    """
    ssim_scores_testing = []

    for batch in testing_ds_batched:
        for n,image in enumerate(batch):
            test_image = image[tf.newaxis, :]

            result = vq_vae_model_overall(test_image, training=False)

            #z = encoder(test_image, training=False)
            #Get the quantized latent space
            #z_quantized, z1 = quantizer_layer(z, training=False)
            #Get the reconstructions
            #result = decoder(z_quantized, training=False)

            ssim = tf.image.ssim(test_image, result, max_val = 1)
            ssim_scores_testing.append(ssim)

            # Show an example every 100th image
            if n % 100 == 0:
                #Plot the resulting image compared to original
                plt.subplot(1,2,1)
                plt.imshow(test_image[0, :, :, 0], cmap='gray')

                plt.subplot(1,2,2)
                plt.imshow(result[0, :, :, 0], cmap='gray')
                plt.show()
                print(ssim.numpy())

    #Display the mean ssim
    ssim_score = (tf.reduce_mean(ssim_scores_testing)).numpy()
    return ssim_score


## Create a pixelCNN and train it for novel image generation
# The subsequent code is from https://keras.io/examples/generative/pixelcnn/ with minor modifications
# A custom callback in the pixelCNN training was added to generate an example novel image with each epoch end

class PixelConvLayer(tf.keras.layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = tf.keras.layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


# Next, we build our residual block layer.
# This is just a normal residual block, but based on the PixelConvLayer.
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return tf.keras.layers.add([inputs, x])


def create_pixelCNN(num_residual_blocks, num_pixelcnn_layers):
    """
    Creates and returns a pixelCNN model object
    """
    pixelcnn_inputs = tf.keras.Input(shape=PIXELCNN_INPUT_SHAPE, dtype=tf.int32)
    one_hot_encoding = tf.one_hot(pixelcnn_inputs, K)
    x = PixelConvLayer(mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same")(one_hot_encoding)

    for _ in range(num_residual_blocks):
        x = ResidualBlock(filters=128)(x)

    for _ in range(num_pixelcnn_layers):
        x = PixelConvLayer(
            mask_type="B",
            filters=128,
            kernel_size=1,
            strides=1,
            activation="relu",
            padding="valid",
        )(x)

    out = tf.keras.layers.Conv2D(filters=K, kernel_size=1, strides=1, padding="valid")(x)
    pixel_cnn = tf.keras.Model(pixelcnn_inputs, out, name="pixel_cnn")
    return pixel_cnn


#PixelCNN Training

def get_codebook_indices(codebook_indices_vector, training_ds, encoder, quantizer_layer):
    """ 
    For the given training dataset, retrieves and returns all the codebook indices.
    Calculates the codebook indices in each batch, by passing through the encoder and quantizer layer.
    Subsequently concatenates each onto a single vector that is returned.
    """
    for n,batch in enumerate(training_ds):
        # Generate the codebook indices.
        encoded_outputs = encoder.predict(batch)
        flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
        # Quantize the encoded vector outputs
        codebook_indices, vectors = quantizer_layer.quantize_vectors(flat_enc_outputs)
        codebook_indices2 = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
        
        # Append the condebook indices onto the given overall codebook_indices_vector
        if n == 0:
            codebook_indices_vector = codebook_indices2
        else:
            codebook_indices_vector = tf.concat([codebook_indices_vector, codebook_indices2], 0)
        #print(codebook_indices_vector.shape)
    
    return codebook_indices_vector

 def train_pixel_cnn(pixel_cnn, epochs, training_ds, encoder, quantizer_layer):
    """
    Completes training of the pixelCNN to learn the prior, for the given number of epochs.
    """
    # Create an empty array in which to store all our codebook indices
    codebook_indices_vector = []
    codebook_indices_vector = get_codebook_indices(codebook_indices_vector, training_ds, encoder, quantizer_layer)

    # Compile the pixelCNN model
    pixel_cnn.compile(
        optimizer = tf.keras.optimizers.Adam(3e-4),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = ["accuracy"],
    )

    #Train the pixelCNN for 500 epochs
    pixel_cnn.fit(
        x = codebook_indices_vector,
        y = codebook_indices_vector,
        batch_size = 128,
        epochs = epochs,
        validation_split = 0.1,
    )

def display_generated_images(num_images, pixel_cnn, quantizer_layer, decoder):
    """
    Produces a final display of the pixelCNN training results.
    Displays 10 generated images in addition to their corresponding latent codes as created by the model.
    """
    batch = num_images
    # Create a blank array with which to feed into the pixelCNN and start the autoregressive prediction.
    priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
    batch, rows, cols = priors.shape

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in range(rows):
        for col in range(cols):
            # Feed the whole array and retrieving the pixel value probabilities for the next pixel.        
            x = pixel_cnn(priors, training=False)
            dist = tfp.distributions.Categorical(logits=x)
            sampled = dist.sample()
            
            # Use the probabilities to pick pixel values and append the values to the priors.
            priors[:, row, col] = sampled[:, row, col]

    # Perform an embedding lookup.
    pretrained_embeddings = quantizer_layer._embeddings
    priors_ohe = tf.one_hot(priors.astype("int32"), K).numpy()
    quantized = tf.matmul(priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))

    # Generate the novel images.
    generated_samples = decoder.predict(quantized)

    for i in range(batch):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i])
        plt.title("Code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated_samples[i].squeeze() + 0.5)
        plt.title("Generated Sample")
        plt.axis("off")
        plt.show()