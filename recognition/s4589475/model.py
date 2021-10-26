#VQ-VAE parameters from the paper
beta = 0.25 #commitment loss weighting - based on the paper
K = 512 #Number of codebook vectors / embeddings number - using recommended value in the paper

# Vector Quantizer Layer
class Vector_Quantizer(tf.keras.layers.Layer):
    
    def __init__(self):
        super().__init__()
        #define the size of our embeddings - has same length as latent space vectors, and width as defined by K (hyperparameter)
        codebook_shape = (latent_dims, K)
        #create the embedding object - initialise uniformly
        initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
        #Declare our embeddings as a tensorflow variable that will be learnt
        self._embeddings = tf.Variable(initializer(shape=codebook_shape, dtype='float32'), trainable = True)

    def quantize_vectors(self, vectors):
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

# Define the encoder and decoder networks
def encoder_network():

    # Encoder network - takes input images and encodes them into z
    inputs = tf.keras.layers.Input(shape=(image_height, image_width, channels))
    
    # First downsample by half
    conv = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation='relu', padding = "same")(inputs)
    #Second downsample by half
    conv = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, activation='relu', padding = "same")(conv)
    #Third downsample by half
    #conv = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, activation='relu', padding = "same")(conv)
    
    #Final conv layer - image is now encoded into size of latent space
    output = tf.keras.layers.Conv2D(latent_dims, kernel_size = 1, padding = "same")(conv) #******************

    # Build the encoder
    final_network = tf.keras.Model(inputs, output, name="Encoder")

    return final_network

def decoder_network():
    # Takes the latent space and upsamples until original size is reached
    #256 -> downsampled twice = 256/4 = 64
    inputs = tf.keras.layers.Input(shape=(64,64,latent_dims))
    #input_shape = encoder().output.shape[1:]
    #inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Now can upsample twice
    conv = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    conv = tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation='relu')(conv)
    
    #Output layer
    output = tf.keras.layers.Conv2D(1,kernel_size=[1,1], strides=1, activation='relu', padding = "same")(conv)
    
    #build the decoder
    final_network = tf.keras.Model(inputs, output, name="Decoder")
    return final_network

def reconstruction_loss(inputs, outputs):  
    #Calculate mse loss
    mse = tf.keras.losses.MeanSquaredError()
    recon_loss = mse(inputs, outputs)
    
    #Calculate ssim to use as part reconstruction loss - focusses on the brain rather than background
    ssim_loss = 1 - tf.image.ssim(inputs, outputs, max_val = 1)
    
    return recon_loss + ssim_loss

#Replace kl-divergence in traditional VAE with two terms as kl-divergence can't be minimized (is a constant)
def latent_loss(encoder_output, quantized_vectors):
    # latent commitment loss
    # beta*||z_e(x)-sg[e]|| i.e. freeze the codebook vectors/embeddings and push the encoded vectors towards codebook vectors
    mse1 = tf.keras.losses.MeanSquaredError()
    e_loss = beta * mse1(tf.stop_gradient(quantized_vectors), encoder_output)

    # latent codebook loss
    #||sg[z_e(x)]-e|| i.e. freeze the encoded vectors and push codebook vectors to encoded vectors
    mse2 = tf.keras.losses.MeanSquaredError()
    q_loss = mse2(quantized_vectors, tf.stop_gradient(encoder_output))
    
    return q_loss + e_loss
#Create overall vqvae model
def create_overall_vqvae(encoder_net, quantizer, decoder_net):
    inputs = tf.keras.Input(shape=(image_height, image_width, channels))
    encoder_outputs = encoder_net(inputs)
    quantized_latents, z1 = quantizer(encoder_outputs)
    reconstructions = decoder_net(quantized_latents)
    vq_vae_model_overall = tf.keras.Model(inputs, reconstructions, name="overall model")
    
    return vq_vae_model_overall


# create a training step
@tf.function
def training_step(images, optimizer, encoder, quantizer_layer, decoder, vq_vae_model_overall):
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
        latent_loss1 = tf.reduce_mean(latent_loss(z, z1))
        
        total_loss = recon_loss + latent_loss1    
    
    gradients = vae_tape.gradient(total_loss, vq_vae_model_overall.trainable_variables)
    
    #apply the gradients to the optimizer
    optimizer.apply_gradients(zip(gradients, vq_vae_model_overall.trainable_variables))
    
    return recon_loss, latent_loss1, total_loss

#Create the encoder and decoder components of VQ-VAE Model
encoder = encoder_network()
decoder = decoder_network()
quantizer_layer = Vector_Quantizer()
optimizer = tf.keras.optimizers.Adam(1e-4)

vq_vae_overall = create_overall_vqvae(encoder, quantizer_layer, decoder)

# Training
def train(encoder_net, decoder_net, quantizer_net, opt, vq_model):
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

            recon_loss, latent_loss_results, total_loss = training_step(image_batch, opt, encoder_net, quantizer_net, decoder_net, vq_model)

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
    
    #return encoder, decoder, quantizer_layer, vq_vae_overall

train(encoder, decoder, quantizer_layer, optimizer, vq_vae_overall)

# Calculate SSIM for the entire testing dataset
def calculate_ssims(vq_vae_model_overall):
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

ssim_score = calculate_ssims(vq_vae_overall)
#Display the mean ssim
print("Average SSIM on the test dataset:"+ (tf.reduce_mean(ssim_scores_testing)).numpy())



### GENERATING NOVEL IMAGES USING OFF-THE-SHELF PIXEL CNN

# PixelCNN
num_residual_blocks = 2
num_pixelcnn_layers = 2

pixelcnn_input_shape = (64,64)


# The first layer is the PixelCNN layer. This layer simply
# builds on the 2D convolutional layer, but includes masking.
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


def create_pixelCNN():
    pixelcnn_inputs = tf.keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
    one_hot_encoding = tf.one_hot(pixelcnn_inputs, K)
    x = PixelConvLayer(
        mask_type="A", filters=128, kernel_size=32, activation="relu", padding="same" #kernel_size = 7!
    )(one_hot_encoding)

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

    out = tf.keras.layers.Conv2D(
        filters=K, kernel_size=1, strides=1, padding="valid"
    )(x)

    pixel_cnn = tf.keras.Model(pixelcnn_inputs, out, name="pixel_cnn")
    return pixel_cnn

pixel_cnn = create_pixelCNN()
pixel_cnn.summary()

#PixelCNN Training

codebook_indices_vector = []

def get_codebook_indices(codebook_indices_vector):
    for n,batch in enumerate(training_ds):
        # Generate the codebook indices.
        encoded_outputs = encoder.predict(batch)
        flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
        codebook_indices, vectors = quantizer_layer.quantize_vectors(flat_enc_outputs)

        codebook_indices2 = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

        if n == 0:
            codebook_indices_vector = codebook_indices2
        else:
            codebook_indices_vector = tf.concat([codebook_indices_vector, codebook_indices2], 0)
        #print(codebook_indices_vector.shape)
    
    return codebook_indices_vector
    
codebook_indices_vector = get_codebook_indices(codebook_indices_vector)
print(f"Shape of the training data for PixelCNN: {codebook_indices_vector.shape}")

pixel_cnn.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
pixel_cnn.fit(
    x=codebook_indices_vector,
    y=codebook_indices_vector,
    batch_size=128,
    epochs=30,
    validation_split=0.1,
)

#sample distinct codes from the pixelcnn output and pass them to the vqvae decoder to generate novel images
# Create a mini sampler model.
inputs = tf.keras.layers.Input(shape=pixel_cnn.input_shape[1:])
x = pixel_cnn(inputs, training=False)
dist = tfp.distributions.Categorical(logits=x)
sampled = dist.sample()
sampler = tf.keras.Model(inputs, sampled)


# Create an empty array of priors.
batch = 10
priors = tf.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
batch, rows, cols = priors.shape

# Iterate over the priors because generation has to be done sequentially pixel by pixel.
for row in range(rows):
    for col in range(cols):
        # Feed the whole array and retrieving the pixel value probabilities for the next
        # pixel.
        probs = sampler.predict(priors)
        # Use the probabilities to pick pixel values and append the values to the priors.
        priors[:, row, col] = probs[:, row, col]

print(f"Prior shape: {priors.shape}")

# Perform an embedding lookup.
pretrained_embeddings = quantizer._embeddings
priors_ohe = tf.one_hot(priors.astype("int32"), K).numpy()
quantized = tf.matmul(
    priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
)
quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))

# Generate novel images.
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