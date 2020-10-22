import tensorflow as tf
import sys
from DataUtils import load_data
import numpy as np


### Define Model:

## Generator:

def generator(input_dim, latent_dim):
    
    input_layer = tf.keras.Input(latent_dim)

    net = tf.keras.layers.Dense(1024)(input_layer)
    net = tf.keras.layers.Dense(input_dim*input_dim*64)(input_layer)
    net = tf.keras.layers.Reshape((input_dim,input_dim,64))(net)
    net = tf.keras.layers.ReLU()(net)

    net = tf.keras.layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(net)
    net = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), padding='same')(net)
    net = tf.keras.layers.ReLU()(net)

    net = tf.keras.layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same')(net)
    net = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same')(net)
    net = tf.keras.layers.ReLU()(net)

    net = tf.keras.layers.Conv2DTranspose(16, (3,3), strides=(2,2), padding='same')(net)
    net = tf.keras.layers.Conv2D(16, (3,3), strides=(1,1), padding='same')(net)
    net = tf.keras.layers.ReLU()(net)

    net = tf.keras.layers.Conv2D(3, (3,3), strides=(1,1), padding='same')(net)

    model = tf.keras.Model(inputs=input_layer, outputs=net)

    return model


def discriminator(input_dim):

    input_layer = tf.keras.Input(input_dim)
    net = input_layer

    net = tf.keras.layers.Conv2D(16, (3,3), padding='same')(net)
    net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
    net = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2))(net)

    net = tf.keras.layers.Conv2D(32, (3,3), padding='same')(net)
    net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
    net = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2))(net)

    net = tf.keras.layers.Conv2D(64, (3,3), padding='same')(net)
    net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
    net = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2))(net)
    
    net = tf.keras.layers.Flatten()(net)

    net = tf.keras.layers.Dense(256)(net)
    net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)

    net = tf.keras.layers.Dense(1)(net)

    model = tf.keras.Model(inputs=input_layer, outputs=net)

    return model

    

### Losses

def discriminator_loss(fake_outputs, real_outputs, batch_size):
    discriminator_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_outputs, labels=tf.ones((batch_size, 1))))
    discriminator_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_outputs, labels=tf.zeros((batch_size, 1))))
    total_discriminator_loss = discriminator_real + discriminator_fake
    return total_discriminator_loss

def generator_loss(fake_outputs, batch_size):
    generator_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_outputs, labels=tf.ones((batch_size, 1))))
    return generator_fake



### Main Training Function

def train(filepath, output_dir, epochs=30, batch_size=128, latent_dim=256, generator_input_dim=8, learning_rate_generator=0.0002, learning_rate_discriminator=0.0002):

    ### Load Data
    image_iter, discriminator_input_dim, dataset_size = load_data(filepath, batch_size)


    ### Construct Models
    disc = discriminator(discriminator_input_dim)
    gen = generator(generator_input_dim, latent_dim)


    ### Define Optimisers
    discriminator_optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate_discriminator, beta_1=0.5)
    generator_optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate_generator, beta_1=0.5)


    ### Training
    gen_hist = list()
    disc_hist = list()

    for i in range(epochs):        
        data = image_iter.get_next()

        gen_hist_temp = list()
        disc_hist_temp = list()

        for k in range(5):
            tf.keras.preprocessing.image.save_img(output_dir + 'TrainImages/Epoch{}_{}.png'.format(i, k), data[k])


        for j in range(dataset_size//batch_size):
            # Obtain data for this batch:
            latent_data = tf.random.normal(shape=(batch_size, latent_dim))
            image_data = image_iter.get_next()

            # Train Discriminator
            with tf.GradientTape() as tape:
                fake_image = gen(latent_data)
                fake_output = disc(fake_image)
                real_output = disc(image_data)
                disc_loss = discriminator_loss(fake_output, real_output, batch_size)
            gradients = tape.gradient(disc_loss, disc.trainable_variables)
            discriminator_optimiser.apply_gradients(zip(gradients, disc.trainable_variables))
            disc_hist_temp.append(disc_loss)

            # Train Generator
            with tf.GradientTape() as tape:
                fake_image = gen(latent_data)
                fake_output = disc(fake_image)
                gen_loss = generator_loss(fake_output, batch_size)
            gradients = tape.gradient(gen_loss, gen.trainable_variables)
            generator_optimiser.apply_gradients(zip(gradients, gen.trainable_variables))
            gen_hist_temp.append(gen_loss)


        # Print output
        print("Epoch: {}".format(i))
        print("Gen Loss: {:.5f}".format(tf.reduce_mean(gen_loss).numpy()))
        print("Disc Loss: {:.5f}".format(tf.reduce_mean(disc_loss).numpy()))
        sys.stdout.flush()

        # Save example images
        latent_data = tf.random.normal(shape=(5, latent_dim))
        fake_images = gen(latent_data).numpy() * 255
        mins = np.min(fake_images, axis=(1,2,3))[:,None,None,None]
        maxs = np.max(fake_images, axis=(1,2,3))[:,None,None,None]
        fake_images = (fake_images - mins)/(maxs-mins)
        for k in range(5):
            tf.keras.preprocessing.image.save_img(output_dir + 'Intermediate/Epoch{}_{}.png'.format(i, k), fake_images[k])

        # Save Model
        gen.save(output_dir + "Models/gen{}.h5".format(i))
        disc.save(output_dir + "Models/disc{}.h5".format(i))

        gen_hist.append(np.mean(gen_hist_temp))
        disc_hist.append(np.mean(disc_hist_temp))

    print("Training Complete")

    return gen_hist, disc_hist