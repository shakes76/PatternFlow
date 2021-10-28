from matplotlib.pyplot import xcorr
import tensorflow as tf
import tensorflow.keras as tfk

class VectorQuantizer(tfk.layers.Layer):
    def __init__(self, n_embeddings, embedding_dim, beta, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.beta = beta 
        self.embeddings = tf.Variable(
            initial_value = tf.random_uniform_initializer()(
                shape=(self.n_embeddings, self.embedding_dim), dtype='float32'
            ),
            trainable=True,
            name='codebook_vectors'
        )

    def call(self, x):
        '''
            This function takes the (HxWxC) encoded image then maps each C-dim pixel vector to the closest embedding C-dim vector defined in the codebook
        '''
        input_shape = tf.shape(x)
        pixel_vectors = tf.reshape(x, [-1, self.embedding_dim]) # The shape of the input now becomes (H*W, C)
        
        #### Quantization - Replace each pixel vector by its closest embedding in the codebook
        # Get the code of the closest vector in the codebook for each pixel vector
        quantized_codes = self.get_closest_codes(pixel_vectors) # The shape of quantized_codes should be (H*W, 1)
        one_hot_encoddings = tf.one_hot(quantized_codes, self.n_embeddings) # Express each discrete code as a one-hot vector -> (H*W, C)
        # Multiply the one-hot row vectors with the embedding matrix will result in another matrix of embeddings,
        # where each row now contains the closest embedding relative to the original pixel vector   
        quantized_x = tf.matmul(one_hot_encoddings, self.embeddings) # (H*W, C)
        quantized_x = tf.reshape(quantized_x, input_shape) # (H, W, C)
        
        #### Add the loss for optimizing the embeddings in the codebook, which contains the commitment loss and the codebook loss
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized_x) - x) ** 2)
        codebook_loss = tf.reduce_mean((tf.stop_gradient(x) - quantized_x) ** 2)
        self.add_loss((self.beta * commitment_loss) + codebook_loss)
        # Apply STE for estimating the gradients of the encoding part of the encoder
        quantized_x = x + tf.stop_gradient(quantized_x - x) # Doing this can make the gradient of this whole quantization process become 1
        return quantized_x
    
    def get_closest_codes(self, pixel_vectors):
        '''
            This function finds and returns the indices of the closest embeddings in the codebook for the given pixel vectors
        '''
        # Get the closest embedding via computing L2 distance between itself and the pixel vector
        # Mathematically, it is (x-y)^T(x-y), where x and y are the vectors of C dimensions
        square_pixel_vector_lens = tf.reduce_sum(pixel_vectors ** 2, axis=1, keepdims=True) # (H*W, 1)
        square_embedding_lens = tf.transpose(tf.reduce_sum(self.embeddings ** 2, axis=1, keepdims=True)) # (1, n_embeddings)
        pixel_vec_dot_embedding = tf.matmul(pixel_vectors, self.embeddings, transpose_b=True) # (H*W, C) x (C, n_embeddings) -> (H*W, n_embeddings)
        distances = square_pixel_vector_lens + square_embedding_lens - (2*pixel_vec_dot_embedding) # (H*W, n_embeddings)
        # Get the index of the closest embedding relative to the pixel vector
        quantized_codes = tf.argmin(distances, axis=1) # (H*W, 1)
        return quantized_codes


def createResidualBlock(inputs, n_channels, latent_kernel_size, strides, use_1x1conv):
    '''
        This function defines an identity residual block, which contains the following layers in order:
        1. A convolutional layer with kernel size = latent_kernel_size and followed by Batch Norm & ReLU
        2. Another convolutional layer with same kernel size and then apply Batch Norm
        3. Add the original input to the output produced in the 2nd step if use_1x1conv is set to False.
            Otherwise, apply a 1x1 conv layer to the input, 
            then add it to the output produced in the 2nd step
        4. Apply ReLU and return the output
    '''
    if strides >= 2:
        assert use_1x1conv, 'If number of strides is greater than one, use_1x1conv must be set to True'
    x = tfk.layers.Conv2D(filters=n_channels, kernel_size=latent_kernel_size, strides=strides, padding='same')(inputs)
    x = tfk.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = tfk.layers.Conv2D(filters=n_channels, kernel_size=latent_kernel_size, padding='same')(x)
    x = tfk.layers.BatchNormalization()(x)
    z = None
    if use_1x1conv:
        y = tfk.layers.Conv2D(filters=n_channels, kernel_size=1, strides=strides)(inputs)
        z = tfk.layers.add([x, y])
    else:
        z = tfk.layers.add([x, inputs])
    return tf.nn.relu(z)

class VQ_VAE(tfk.Model):
    def __init__(self, img_h, img_w, img_c, train_variance, embedding_dim, n_embeddings, recon_loss_type, commitment_factor, **kwargs):
        super(VQ_VAE, self).__init__(**kwargs)
        self.img_h = img_h # Height of an input image
        self.img_w = img_w # Width of an input image
        self.img_c = img_c # Number of channels of an input image
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.train_variance = train_variance
        self.recon_loss_type = recon_loss_type
        self.commitment_factor = commitment_factor
        self.vq_vae = self.create_vq_vae()
        self.total_loss_tracker = tfk.metrics.Mean(name='total_loss')
        self.vq_loss_tracker = tfk.metrics.Mean(name='vq_loss')
        self.recon_loss_tracker = tfk.metrics.Mean(name='recon_loss')
        self.val_ssim_tracker = tfk.metrics.Mean(name='SSIM')
        self.val_mse_tracker = tfk.metrics.Mean(name='MSE')

    def create_encoder(self):
        '''
            This function creates the encoder part of the VQ-VAE and returns the encoder as well as the height and the width of an encoded image
        '''
        inputs = tfk.Input(shape=(self.img_h, self.img_w, self.img_c), name='encoder_input')
        ## First CNN block
        x = tfk.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(inputs) # 88x88
        x = tfk.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        ## Second CNN block
        x = tfk.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(x) # 44x44
        x = tfk.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        ## Third CNN block
        x = tfk.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(x) # 22x22
        x = tfk.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        ## Residual blocks
        x = createResidualBlock(x, 64, 3, 1, True) # 22x22
        x = createResidualBlock(x, 64, 3, 1, False)
        x = createResidualBlock(x, 64, 3, 1, False)
        x = createResidualBlock(x, 64, 3, 1, False)
        x = createResidualBlock(x, 64, 3, 1, False)
        # Pre-quantized layer
        x = tfk.layers.Conv2D(filters=self.embedding_dim, kernel_size=1, strides=1)(x)
        x = tfk.layers.BatchNormalization()(x)
        # Extract the height and the width of an encoded image
        encoder_shape = tfk.backend.int_shape(x)
        encoder_h, encoder_w = encoder_shape[1], encoder_shape[2]
        # Create model
        encoder = tfk.Model(inputs, x, name='encoder')
        return encoder, encoder_h, encoder_w

    def create_decoder(self, encoder_h, encoder_w):
        '''
            This function creates the decoder part of the VQ-VAE and returns the decoder, where the input shape of the decoder
            corresponds to the height, the width, and the number of channels == embedding size of an encoded image
        '''
        inputs = tfk.Input(shape=(encoder_h, encoder_w, self.embedding_dim))
        # Post-quantized layer
        x = tfk.layers.BatchNormalization()(inputs)
        x = tfk.layers.Conv2D(filters=64, kernel_size=1, strides=1)(x)
        # Residual blocks
        x = createResidualBlock(x, 64, 3, 1, False)
        x = createResidualBlock(x, 64, 3, 1, False)
        x = createResidualBlock(x, 64, 3, 1, False)
        x = createResidualBlock(x, 64, 3, 1, False)
        x = createResidualBlock(x, 256, 3, 1, True)
        ## First CNN block
        x = tfk.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x) # 22x22
        x = tfk.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        ## Second Transposed CNN block
        x = tfk.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same')(x) # 44x44
        x = tfk.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        ## Third Transposed CNN block
        x = tfk.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(x) # 88x88
        x = tfk.layers.BatchNormalization()(x)
        outputs = tfk.layers.Conv2DTranspose(filters=self.img_c, kernel_size=3, strides=2, padding='same')(x)
        decoder = tfk.Model(inputs, outputs, name='decoder')
        return decoder
    
    def create_vq_vae(self):
        quantizer = VectorQuantizer(self.n_embeddings, self.embedding_dim, self.commitment_factor)
        encoder, encoder_h, encoder_w = self.create_encoder()
        #encoder.summary() # Print out the architecture of the encoder
        decoder = self.create_decoder(encoder_h, encoder_w)
        #decoder.summary() # Print out the architecture of the decoder
        inputs = tfk.Input(shape=(self.img_h, self.img_w, self.img_c))
        encoded_imgs = encoder(inputs)
        quantized_imgs = quantizer(encoded_imgs)
        reconstructed_imgs = decoder(quantized_imgs)
        return tfk.Model(inputs, reconstructed_imgs, name='vq_vae')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.vq_loss_tracker
        ]
    
    def train_step(self, imgs):
        with tf.GradientTape() as tape:
            reconstructed_imgs = self.vq_vae(imgs)
            if self.recon_loss_type == 'SSIM':
                # Calculate the reconstruction loss using SSIM
                # The idea is basically taken from https://arxiv.org/pdf/1511.08861.pdf
                recon_loss = 1 - tf.reduce_mean(
                    tf.image.ssim(imgs, reconstructed_imgs, max_val=1.0)
                )
            else:
                recon_loss = tf.reduce_mean(
                    ((imgs - reconstructed_imgs) ** 2) / self.train_variance
                )
            # VQ loss = code book loss + commitment loss
            vq_loss = sum(self.vq_vae.losses)
            # Total loss = recon loss + vq loss
            total_loss = recon_loss + vq_loss
        # Backpropagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Trak the new loss values
        self.total_loss_tracker.update_state(total_loss)
        self.vq_loss_tracker.update_state(vq_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        return {
            'total loss': self.total_loss_tracker.result(),
            'reconstruction loss': self.recon_loss_tracker.result(),
            'VQ loss': self.vq_loss_tracker.result()
        }
    
    def test_step(self, imgs):
        reconstructed_imgs = self.vq_vae(imgs, training=False)
        # Compute the SSIM of validation images
        val_ssim = tf.image.ssim(imgs, reconstructed_imgs, max_val=1.0)
        val_mse = tf.reduce_mean(((imgs - reconstructed_imgs) ** 2) / self.train_variance, axis=(1, 2, 3))
        self.val_ssim_tracker.update_state(val_ssim)
        self.val_mse_tracker.update_state(val_mse)
        return {
            'SSIM': self.val_ssim_tracker.result(),
            'MSE': self.val_mse_tracker.result()
        }