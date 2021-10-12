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
    
    #Flatten for the Dense layers
    flat = tf.keras.layers.Flatten()(conv)
    dense = tf.keras.layers.Dense(256, activation="relu")(flat)
    dense = tf.keras.layers.Dense(128, activation="relu")(dense)

    #Final dense layer - image is now encoded into size of hidden space
    hidden = tf.keras.layers.Dense(latent_dims)(dense)

    # Convert this latent representation to a quantized vector
    output = quantized(hidden)
    
    # Build the encoder
    final_network = tf.keras.Model(inputs, output)

    return final_network
    
    
#def encoder_loss():


def decoder():
    # Takes the latent space and upsamples until original size is reached
    #256 -> downsampled twice = 256/4 = 64
    inputs = tf.keras.layers.Input(shape=(latent_dims,))
    
    dense = tf.keras.layers.Dense(128, activation = "relu")(inputs)
    dense = tf.keras.layers.Dense(256, activation = "relu")(dense)

    dense = tf.keras.layers.Dense(64*64*depth*2, activation = "relu")(dense)

    # Reshape back into 2D for the conv layers
    dense = tf.keras.layers.Reshape((64,64,depth*2))(dense)
    
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
