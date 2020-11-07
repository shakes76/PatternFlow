import tensorflow as tf
from tensorflow.keras import layers

# time is used to time the epochs. 
# matplotlib is used to display the fake brain images and figures
import time
import matplotlib.pyplot as plt

# Allows more of the GPU's memory to be used
tf.__version__
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Batch size of 4 is the maximum we can set with our network complexity
# without overloading the GPU's memory
BATCH_SIZE = 4

noise_dim = 100 # How many noise values we use 
smooth = 0.8 # Smoothing parameter attempts to stop the Generator/Discriminator from becoming too powerful (loss converge to 0 too fast).
EPOCHS = 40 # More epochs that this do not yield better results

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5) # learing rate and beta_1 set to avoid oscilating convergence
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

gen_loss = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
disc_loss = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ssim = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

# Generator model upsamples from a 4x4x1024 network to a 256x256x1 tensor. 
# Each time we increase the resolution, we have to half the depth/number of filters so as not to overload the GPU memory.
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4 * 4 * 1024, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 1024)))

    model.add(layers.Conv2DTranspose(512, (1, 1), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

# Generator takes input of 256x256x1, downsamples through convolution and outputs single value between 0 and 1.
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[256, 256, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

generator = make_generator_model()

discriminator = make_discriminator_model()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output * smooth), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output * smooth), fake_output)

# Using tf.function here greatly increases the speed of the algorithm. 
@tf.function
def train_step(images):
    
    # We use a standard normal distribution for our generators input
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    # Performs the main training loop, training the parameters with each batch
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    

# Performs the training loop for each batch in each epoch.
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        i = 0;
        for image_batch in dataset:
            train_step(image_batch)
            
            # Outputs a generated image and other information at the beginning of each epoch.
            if (i == 0):
               seed = tf.random.normal([BATCH_SIZE, noise_dim])
               generate_image(generator, epoch + 1, seed, dataset, image_batch)
               i = i + 1
               
        # Outputs the time for each epoch.       
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


def generate_image(model, epoch, test_input, dataset, images):
    predictions = model(test_input, training=False)
    
    ssimmax = 0
    i = 0
    for image_batch in dataset:
            ssim1 = tf.image.ssim(predictions, image_batch, 2)
            ssimmax = max(ssimmax, tf.math.reduce_max(ssim1))
            i = i + 1;
            if(i >= 7):
                break
            
                    
    
    
    real_output = discriminator(images, training=False)
    fake_output = discriminator(predictions, training=False)

    gen_loss[epoch - 1] = generator_loss(fake_output)
    disc_loss[epoch - 1] = discriminator_loss(real_output, fake_output)
    ssim[epoch - 1] = ssimmax

    print('Generator Loss {}      Discriminator Loss {}'.format(gen_loss[epoch - 1], disc_loss[epoch - 1]))

    print("SSIM: {}".format(tf.math.reduce_mean(ssimmax)))
               
    plt.imshow(predictions[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')

    plt.show()
    
def run_GAN(dataset):
    
    train(dataset, EPOCHS)
    plt.plot(epochs, gen_loss, label = 'Generator Loss')
    plt.plot(epochs, disc_loss, label = 'Discriminator Loss')
    plt.plot(epochs, ssim, label = 'SSIM')
    plt.xlabel('EPOCHS')
    plt.legend(loc = "upper right")
    plt.show()
