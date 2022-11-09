import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import sys

def DCGAN(train_dir, test_dir, result, epochs=10000, batch_size=256):
    train_size = 9665
    test_size = 545
    img_size = (128, 128, 1)

    # get the filenames of all images + split them into training and testing sets
    # uses os to load images as permitted by instructor answer on post 208 on piazza 
    train_image_names = os.listdir(train_dir)
    test_image_names = os.listdir(test_dir)

    #print("Filenames collected")

    def get_images(img_dir, img_names):
        """
        Get all files from img_dir with names in img_names.
        
        Uses numpy to load and preprocess images as permitted by instructor answer 
        on post 208 on piazza
        """
        images = []
        end_count = len(img_names)
        for i, name in enumerate(img_names):
            if ".png" not in name and ".jpg" not in name:
                # the name represents a file that is not an image
                continue
            # print/update a progress bar (without this it looks like nothing is 
            # happening which can be frustrating
            print("{:.0f}% of files gathered".format(i*100/end_count), end="\r")
            image = load_img(img_dir + name, target_size=img_size, 
                             color_mode="grayscale")
            # convert to array and normalise
            image = img_to_array(image)/255.0
            images.append(image)
        return np.array(images)

    # get training set
    print("TRAINING SET")
    X_train = get_images(train_dir, train_image_names)
    print("\n")

    # get testing set
    print("TESTING SET")
    X_test = get_images(test_dir, test_image_names)
    print("\n")

    def build_generator(noise_shape=(100,)):
        """
        Builds a generator based on a given input shape, noise_shape
        """
        input_noise = layers.Input(shape=noise_shape)

        l = layers.Dense(128*8*8, activation="relu")(input_noise)
        l = layers.Reshape((8, 8, 128))(l)
        
        l = layers.Conv2DTranspose(128, kernel_size=(2, 2), strides=(2,2), 
                                   use_bias=False)(l)
        l = layers.Conv2D(128, (1, 1), activation="relu", padding="same")(l)
        l = layers.BatchNormalization()(l)
        
        l = layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2,2),
                                   use_bias=False)(l)
        l = layers.Conv2D(64 , (1, 1), activation="relu", padding="same")(l)
        l = layers.BatchNormalization()(l)
        
        l = layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2,2), 
                                   use_bias=False)(l)
        l = layers.Conv2D(32 , (1, 1), activation="relu", padding="same")(l)
        l = layers.BatchNormalization()(l)
        
        l = layers.Conv2DTranspose(16, kernel_size=(2, 2), strides=(2,2), 
                                   use_bias=False)(l)
        img = layers.Conv2D(1, (1, 1), activation="tanh", padding="same")(l)
        
        model = models.Model(input_noise, img)

        return model

    # set our optimizer - they all share
    opt = tf.keras.optimizers.Adam(0.0002, 0.5)

    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=opt)

    def get_noise(nsample=1, latent_dim=100):
        """
        Generates an array of random noise with shape (nsample, latent_dim)
        """
        noise = tf.random.normal((nsample, latent_dim))
        return noise

    def plot_generated_images(noise, path_save=None ,title=""):
        """
        Creates a plot of 4 generated images from input noise. If path_save is not 
        None, saves the images to that path
        """
        images = generator.predict(noise)
        fig = plt.figure(figsize=(40,10))
        for i, img in enumerate(images):
            ax = fig.add_subplot(1,4,i+1)
            ax.imshow(img.squeeze(), cmap="gray")
        fig.suptitle("Generated images "+title,fontsize=30)
        
        if path_save is not None:
            plt.savefig(path_save,
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close()
        else:
            plt.show()
            
    nsample = 4
    noise = get_noise(nsample=nsample, latent_dim=100)

    def build_discriminator(image_shape):
        """
        Builds a discriminator based on a given image_shape
        """
        input_img = layers.Input(shape=image_shape)
        
        l = layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2), 
                          padding='same')(input_img)
        l = layers.LeakyReLU()(l)
        l = layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2), padding='same')(l)
        l = layers.LeakyReLU()(l)
        
        l = layers.Conv2D(64, kernel_size=(3, 3), strides=(2,2), padding='same')(l)
        l = layers.LeakyReLU()(l)
        l = layers.Conv2D(64, kernel_size=(3, 3), strides=(2,2), padding='same')(l)
        l = layers.LeakyReLU()(l)
        
        l = layers.Conv2D(128, kernel_size=(3, 3), strides=(2,2), padding='same')(l)
        l = layers.LeakyReLU()(l)
        l = layers.Conv2D(128, kernel_size=(3, 3), strides=(2,2), padding='same')(l)
        l = layers.LeakyReLU()(l)
        
        l = layers.Flatten()(l)
        out = layers.Dense(1, activation='sigmoid')(l)
        model = models.Model(input_img, out)
        
        return model

    discriminator = build_discriminator(img_size)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt, 
                          metrics=['accuracy'])

    noise_in = layers.Input(shape=(100,))
    img = generator(noise_in)

    # right now we only want to train the generator
    discriminator.trainable = False

    # this returns if the generated image is valid or not
    isValid = discriminator(img)

    # the combined model takes noise as input, generates an image and then 
    # determines if it looks like a brain
    combo = models.Model(noise_in, isValid)
    combo.compile(loss="binary_crossentropy", optimizer=opt)

    def train(models, X_train, noise, result, 
              epochs=10000, batch_size=256):
        """
        Trains the DCGAN on X_train and saves 
        """
        combo, disc, gen = models
        noise_latent = noise.shape[1]
        half = int(batch_size / 2)
        history = []
        
        for epoch in range(epochs):
            # start with discriminator
            
            # select half the images
            indices = tf.random.uniform([half], 0, X_train.shape[0], 
                                        dtype=tf.dtypes.int32)
            images = X_train[indices]
            disc_noise = get_noise(half, noise_latent)
            
            # generate some new images
            gen_images = gen.predict(disc_noise)
            
            # train the discriminator!!
            disc_loss_real = disc.train_on_batch(images, tf.ones((half, 1)))
            disc_loss_fake = disc.train_on_batch(gen_images, tf.zeros((half, 1)))
            disc_loss = 0.5 * tf.math.add(disc_loss_real, disc_loss_fake)
            
            # train generator
            gen_noise = get_noise(batch_size, noise_latent)
            
            # we want all generated images to be labeled as valid
            valid = tf.reshape((tf.convert_to_tensor([1]*batch_size)), 
                               (batch_size, 1))
            
            combo_loss = combo.train_on_batch(gen_noise, valid)
            
            history.append({"D":disc_loss[0], "G":combo_loss})
            
            # Plot the progress
            if epoch % 500 == 0:
                plot_generated_images(noise,
                                      path_save=result+"image_{:05.0f}.png".
                                      format(epoch), 
                                      title="Epoch {}".format(epoch))
                print(("Epoch {:05.0f} [D loss: {:4.3f}, acc.: {:05.1f}%]"+
                        "[G loss: {:4.3f}]").format(epoch, disc_loss[0], 
                       100*disc_loss[1], combo_loss))
            else:
                print(("Epoch {:05.0f} [D loss: {:4.3f}, acc.: {:05.1f}%]"+
                        "[G loss: {:4.3f}]").format(epoch, disc_loss[0], 
                       100*disc_loss[1], combo_loss), end="\r")
        plot_generated_images(noise, path_save=result+"image_final.png".
                              format(epoch), title="Final Run".format(epoch))
        print(("Epoch {:05.0f} [D loss: {:4.3f}, acc.: {:05.1f}%]"+
                        "[G loss: {:4.3f}]").format(epoch, disc_loss[0], 
                       100*disc_loss[1], combo_loss))
        return history
        
    noise = get_noise(nsample=4, latent_dim=100)
    history = train((combo, discriminator, generator), X_train, noise=noise, 
                    result=result)
    
