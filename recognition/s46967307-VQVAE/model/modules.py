import tensorflow as tf
import numpy as np

# Hyperparams
latent_dims = 128
image_shape = (256, 256, 1)
num_embeddings = 256
encoder_depth = 4
encoded_image_shape = (int(256/pow(2,encoder_depth)), int(256/pow(2,encoder_depth)), int(latent_dims))
pixelcnn_input_shape = (int(256/pow(2,encoder_depth)), int(256/pow(2,encoder_depth)), int(1))
beta = 4.0

def ssim_loss(x, y):
    """
    ssim_loss
    args: x - the original set of images
          y - the images being compared
    compares images in x and y element-wise and calculates SSIM
    returns the reduced mean for loss
    """
    return tf.reduce_mean(tf.image.ssim(x,y,1.0))

class PixelConvLayer(tf.keras.layers.Layer):
    """
    PixelConvLayer
    The convolutional layer unique to pixelcnn which implements masking of 
    future pixels, and the current if type is A
    otherwise functions as a normal conv layer.
    """

    def __init__(self, mask_type, stack = 'VH', **kwargs):
        """
        args: mask_type - 'A' or 'B'. 'A' masks the central pixel, 'B' does not
              stack     - 'Whether the conv layer checks horizontal, vertical or both
                          as 'V', 'VH', or 'H'
        """
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.stack = stack
        self.conv = tf.keras.layers.Conv2D(**kwargs)
    
    def build(self, input_shape):
        self.conv.build(input_shape)

        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        if self.stack in ['V', 'VH']:
            self.mask[:kernel_shape[0] // 2, ...] = 1.0 # Sets points above center to 1
        if self.stack in ['H', 'VH']:
            self.mask[kernel_shape[0] // 2, :kernel_shape[1] // 2, ...] = 1.0 # Sets points left of center to 0
        self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 0.0 # Just in case, set the middle to 0

        # Pixelcnn uses 3 channels on mask B, but we only have grayscale data
        # Only change is to include centre pixel in mask
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0
    
    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

def get_pixel_cnn(kernel_size, input_shape):
    """
    get_pixel_cnn
    args: kernel_size - the size of the kernel for the first 'A' layer
          input_shape - the shape for the input layer
    returns: a pixelcnn with the provided kernel size and input shape
    """
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.int32)
    onehot = tf.one_hot(inputs, num_embeddings)
    onehot = tf.keras.layers.Dropout(0.2)(onehot)
    x1 = PixelConvLayer(
        mask_type="A",
        stack='V',
        filters=128,
        kernel_size=kernel_size,
        padding="same")(onehot)
    x2 = PixelConvLayer(
        mask_type="A",
        stack='H',
        filters=128,
        kernel_size=kernel_size,
        padding="same")(onehot)
    x = tf.keras.layers.Add()([x1, x2])
    
    # Residual block layers
    for _ in range(2):
        y = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=1,
        )(x)
        y1 = PixelConvLayer(
            mask_type="B",
            stack='H',
            filters=64,
            kernel_size=3,
            padding="same"
        )(y)
        y2 = PixelConvLayer(
            mask_type="B",
            stack='V',
            filters=64,
            kernel_size=3,
            padding="same"
        )(y)
        y = tf.keras.layers.Add()([y1,y2])
        y = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=1,
        )(y)
        x = tf.keras.layers.Add()([x,y])

    for _ in range(2):
        x1 = PixelConvLayer(
            mask_type="B",
            stack='V',
            filters=256,
            kernel_size=1,
            strides=1,
            padding="valid")(x)
        x2 = PixelConvLayer(
            mask_type="B",
            stack='H',
            filters=256,
            kernel_size=1,
            strides=1,
            padding="valid")(x)
        x = tf.keras.layers.Add()([x1,x2])

    # Flatten each pixel down to the number of embeddings
    x= tf.keras.layers.Conv2D(
        filters=num_embeddings,
        kernel_size=1,
        strides=1,
        padding="valid",
        activation="relu")(x)

    
    return tf.keras.Model(inputs, x)

@tf.function
def outer(y, embeddings):
    """
    returns the inner function over inputs y and each x from the embeddings
    """
    return tf.vectorized_map(
        lambda x:
        inner(x, y),
        embeddings)

@tf.function
def inner(x, y):
    """
    returns the normalized tensor x-y 
    """
    return tf.norm(tf.math.subtract(x, y))

def get_indices(embeddings, inputs_flat, quantize=True, splits=1):
    """
    get_indices
    args: embeddings  - the set of embeddings from the vqvae
          inputs_flat - the flattened set of latent vectors 
          quantize    - if true return the values rather than indexes
          splits      - for large input data this can determine the number
                        of batches to avoid OOM
    """
    split_inputs_flat = tf.split(inputs_flat, splits, axis=0)

    results = None
    for batch in split_inputs_flat:
        batch_results = tf.vectorized_map(
            lambda y:
            outer(y, embeddings),
            batch)
        
        if results is None:
            results = batch_results
        else:
            results = tf.concat([results, batch_results], axis=0)

    results = tf.math.argmin(results, axis=-1)
    if quantize:
        results = tf.matmul(tf.one_hot(
            results, num_embeddings), embeddings)

    return results

class VQ(tf.keras.layers.Layer):
    """
    VQ
    Vector Quantisation layer,
    converts the latent dimension outputted by encoder of VQVAE into
    the nearest vectors in the codebook in VQVAE.
    """
    def __init__(self, **kwargs):
        super(VQ, self).__init__(**kwargs)

        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            trainable=True,
            name="embeddings",
            initial_value=w_init(
                shape=(num_embeddings, latent_dims), dtype="float32")
        )

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        inputs_flat = tf.reshape(inputs, shape=(-1, latent_dims))

        # Get quantized vectors
        results = get_indices(self.embeddings, inputs_flat)

        # Get codebook loss
        codebook_loss = tf.reduce_mean(
            tf.square(tf.stop_gradient(results) - inputs_flat))
        commitment_loss = tf.reduce_mean(
            tf.square(results - tf.stop_gradient(inputs_flat))) * beta
        self.add_loss(commitment_loss + codebook_loss)

        # Reshape results back into compressed image
        results = tf.reshape(results, shape=inputs_shape)

        return inputs + tf.stop_gradient(results - inputs)

class AE(tf.keras.Model):
    """
    AE
    Autoencoder, contains three submodels, the encoder, vq, and deocder
    vq is described in the vq class.
    The encoder reduces the input images by 2^n (see hyperparameter)
    The decoder reverses this effect after vq has been applied
    """

    def __init__(self, **kwargs):
        super(AE, self).__init__(**kwargs)

        # ------ ENCODER -------
        # Takes image as input, runs it through 3 convolutional
        # layers which each halve the size of the image.
        # The remaining image is flattened and shrunk into a
        # latent space.
        input = tf.keras.layers.Input(shape=image_shape, batch_size=None, name="input")
        x = input
        for n in range(encoder_depth):
            x = tf.keras.layers.Conv2D(
                filters = min(128,64*int(pow(2,n))), 
                kernel_size = 3, 
                strides = 2, 
                activation = 'relu',
                padding = "same", 
                name = f"compression_{n}")(x)
        x = tf.keras.layers.Conv2D(
            filters = latent_dims, 
            kernel_size=1,
            strides=1,
            activation='relu',
            padding = "same", 
            name = "to_latent")(x)

        self.encoder = tf.keras.Model(input, x, name="encoder")

        # ------ VQ Layer ------
        # Takes output from encoder.
        # Returns the closest vector in the embedding to the latent
        # space.
        input = tf.keras.layers.Input(shape=encoded_image_shape, batch_size=None, name="input")
        x = VQ(name="vq")(input)
        self.vq = tf.keras.Model(input, x, name="vq")

        # ------ DECODER -------
        # Takes output from VQ layer.
        # Structure is identical to encoder but with Conv2DTranspose
        # to upscale the image rather than downscale.
        input = tf.keras.layers.Input(shape=encoded_image_shape, batch_size=None, name="input")
        x = input
        for n in range(encoder_depth):
            x = tf.keras.layers.Conv2DTranspose(
                filters = min(128,64*int(pow(2,n))), 
                kernel_size = 3, 
                strides = 2, 
                padding = 'same',
                activation = 'relu', 
                name = f"reconstruct_{n}")(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters = 1,
            kernel_size = 1,
            strides = 1,
            padding = 'same',
            name = "to_image",
            activation = 'sigmoid')(x)

        self.decoder = tf.keras.Model(input, x, name="decoder")

    def train_step(self, train_data):
        x, _ = train_data
        with tf.GradientTape() as tape:
            out = self.call(x)

            rc_loss = tf.keras.losses.mean_squared_error(
                x, tf.reshape(out, shape=tf.shape(x)))
            vq_loss = sum(self.vq.losses)

            loss = rc_loss + vq_loss

        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        return {"loss": loss}

    def test_step(self, test_data):
        x, _ = test_data
        out = self.call(x)
        rc_loss = tf.keras.losses.mean_squared_error(
            x, tf.reshape(out, shape=tf.shape(x)))
        vq_loss = sum(self.vq.losses)

        loss = rc_loss + vq_loss

        return {"loss": loss}

    def call(self, inputs):
        return self.decoder(self.vq(self.encoder(inputs)))
