from pickletools import optimize
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

import modules
import dataset
import predict

# The first layer is the PixelCNN layer. This layer simply
# builds on the 2D convolutional layer, but includes masking.
class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

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
class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])


def main():
    #Get data file path locations
    train_images = dataset.FetchData('D:\\University\\2022 Sem 2\\COMP3710\\Project Report\\BranchFOlder\\keras_png_slices_data\\keras_png_slices_data\\keras_png_slices_train\\*')
    test_images = dataset.FetchData('D:\\University\\2022 Sem 2\\COMP3710\\Project Report\\BranchFOlder\\keras_png_slices_data\\keras_png_slices_data\\keras_png_slices_test\\*')
    validate_images = dataset.FetchData('D:\\University\\2022 Sem 2\\COMP3710\\Project Report\\BranchFOlder\\keras_png_slices_data\\keras_png_slices_data\\keras_png_slices_validate\\*')
    #Extract all the images in each file and do some pre-processing
    train = dataset.ImageExtract(train_images)
    test = dataset.ImageExtract(test_images)
    validate = dataset.ImageExtract(validate_images)
    #combine training and validation into one larger set for training
    Oasis = dataset.combine(train, validate)
    #Change test and validate set dimensions for later use
    test = np.squeeze(test)
    test = np.expand_dims(test, -1).astype("float32")
    validate = np.squeeze(validate)
    validate = np.expand_dims(validate, -1).astype("float32")
    #Check to make sure the train and validate sets have been combines
    #check that they have the right shape (256,256) and values between 0 and 1
    print(Oasis.shape)
    print(Oasis.min(), Oasis.max())
    #Check the summaries for the encoder, decoder and combined vqvae models
    modules.new_encoder(32).summary()
    modules.new_decoder(32).summary()
    modules.get_vqvae(32, 128).summary()
    #determine the var in the Oasis set
    variance = np.var(Oasis)
    #build, compile and fit model
    model = modules.VQVAE(variance, latent_dim=32, num_embeddings=128)
    model.compile(optimizer=keras.optimizers.Adam())
    history = model.fit(Oasis, epochs=30, batch_size=128)
    #Show Reconstruction Loss
    plt.subplot(211)
    plt.title('Reconstruction Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.plot(history.history['reconstruction_loss'])

    trained_vqvae_model = modules.model.vqvae
    idx = np.random.choice(len(test), len(test))
    test_images = test[idx]
    reconstructions_test = trained_vqvae_model.predict(test_images)
    simArray = []
    for test_image, recon_img in zip(test_images, reconstructions_test):
        simArray.append(predict.calculate_ssim(test_image, recon_img))
    #Determine the Average ssim over the dataset (test set)
    print(predict.average_ssim(simArray)) 
    trained_vqvae_model = model.vqvae
    idx = np.random.choice(len(test), 15)
    print(idx)
    test_images = test[idx]
    reconstructions_test = trained_vqvae_model.predict(test_images)

    #Show samples of the origional image along with the reconstructed image
    for test_image, reconstructed_image in zip(test_images, reconstructions_test):
        predict.show_subplot(test_image, reconstructed_image)
        sim = predict.calculate_ssim(test_image, reconstructed_image)
        print(sim)

    encoder = model.vqvae.get_layer("encoder")
    quantizer = model.vqvae.get_layer("vector_quantizer")

    encoded_outputs = encoder.predict(test_images)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

    for i in range(len(test_images)):
        plt.subplot(1, 2, 1)
        plt.imshow(test_images[i].squeeze() + 0.5)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(codebook_indices[i])
        plt.title("Code")
        plt.axis("off")
        plt.show()

    num_residual_blocks = 2
    num_pixelcnn_layers = 2
    pixelcnn_input_shape = encoded_outputs.shape[1:-1]
    print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")   

    #Add code to access the pixelCNN model
    pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
    ohe = tf.one_hot(pixelcnn_inputs, model.num_embeddings)
    x = PixelConvLayer(
        mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
    )(ohe)

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

    out = keras.layers.Conv2D(
        filters=model.num_embeddings, kernel_size=1, strides=1, padding="valid"
    )(x)

    pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")
    pixel_cnn.summary()

    # Generate the codebook indices.
    encoded_outputs = encoder.predict(validate)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)

    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
    print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}") 

    pixel_cnn.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    pixel_cnn.fit(
        x=codebook_indices,
        y=codebook_indices,
        batch_size=128,
        epochs=30,
        validation_split=0.1,
    )

    # Create a mini sampler model.
    inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
    outputs = pixel_cnn(inputs, training=False)
    categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
    outputs = categorical_layer(outputs)
    sampler = keras.Model(inputs, outputs)

    # Create an empty array of priors.
    batch = 10
    priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
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
    pretrained_embeddings = quantizer.embeddings
    priors_ohe = tf.one_hot(priors.astype("int32"), model.num_embeddings).numpy()
    quantized = tf.matmul(
        priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
    )
    quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))

    # Generate novel images.
    decoder = model.vqvae.get_layer("decoder")
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


if __name__ == "__main__":
    main()