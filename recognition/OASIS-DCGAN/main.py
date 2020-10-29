import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array

print("Imports done")

train_dir = "H:/keras_png_slices_train/"
test_dir = "H:/keras_png_slices_test/"

train_size = 9665
test_size = 545
img_size = (128, 128, 1)  # 256x256, grayscale

# get the filenames of all the images, then split them into training and testing sets 
import os
train_image_names = os.listdir(train_dir)
test_image_names = os.listdir(test_dir)

print("Filenames collected")

def get_images(img_dir, img_names):
    images = []
    for i, name in enumerate(img_names):
        if name == "Thumbs.db":
            continue
        # print("Getting {}".format(img_dir+name))
        image = load_img(img_dir + name, target_size=img_size, color_mode="grayscale")
        # convert to array and normalise
        image = img_to_array(image)/255.0
        images.append(image)
    return np.array(images)

X_train = get_images(train_dir, train_image_names)
print("X_train.shape =", X_train.shape)

X_test = get_images(test_dir, test_image_names)
print("X_test.shape =", X_test.shape)

print("Images got")

# lookie code: remove for final version
#fig = plt.figure(figsize=(30, 10))
#nplot = 7
#for num in range(1, nplot):
#    ax = fig.add_subplot(1,nplot,num)
#    ax.imshow(X_train[num])
#plt.show()
    
#(X_train, X_test) = 

def build_generator(noise_shape=(100,)):
    input_noise = layers.Input(shape=noise_shape)

    l = layers.Dense(128*8*8, activation="relu")(input_noise)
    l = layers.Reshape((8, 8, 128))(l)
    
    l = layers.Conv2DTranspose(128, kernel_size=(2, 2), strides=(2,2), use_bias=False)(l)
    l = layers.Conv2D(128, (1, 1), activation="relu", padding="same")(l)
    
    l = layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2,2), use_bias=False)(l)
    l = layers.Conv2D(64 , (1, 1), activation="relu", padding="same")(l)
    
    l = layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2,2), use_bias=False)(l)
    l = layers.Conv2D(32 , (1, 1), activation="relu", padding="same")(l)
    
    l = layers.Conv2DTranspose(16, kernel_size=(2, 2), strides=(2,2), use_bias=False)(l)
    l = layers.Conv2D(16 , (1, 1), activation="relu", padding="same")(l)
    
    img = layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(l)
    
    model = models.Model(input_noise, img)

    return model

opt = tf.keras.optimizers.Adam(0.0001, 0.5)

generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=opt)

#generator.summary()

def get_noise(nsample=1, latent_dim=100):
    noise = np.random.normal(0, 1, (nsample, latent_dim))
    #print(noise)
    return noise

def plot_generated_images(noise, path_save=None ,title=""):
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
#plot_generated_images(noise)

def build_discriminator(image_shape, noutput=1):
    input_img = layers.Input(shape=image_shape)
    
    l = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    l = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(l)
    l = layers.MaxPooling2D((2, 2), strides=(2, 2))(l)
    
    l = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(l)
    l = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(l)
    l = layers.MaxPooling2D((2, 2), strides=(2, 2))(l)
    
    l = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(l)
    l = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(l)
    l = layers.MaxPooling2D((2, 2), strides=(1, 1))(l)
    
    l = layers.Flatten()(l)
    l = layers.Dense(1024, activation="relu")(l)
    out = layers.Dense(noutput, activation='sigmoid')(l)
    model = models.Model(input_img, out)
    
    return model

discriminator = build_discriminator(img_size)
discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

#discriminator.summary()

noise_in = layers.Input(shape=(100,))
img = generator(noise_in)
#print(img.shape)

# right now we only want to train the generator
discriminator.trainable = False

# this returns if the generated image is valid or not
isValid = discriminator(img)
#print(isValid.shape)

# the combined model takes noise as input, generates an image and then determines if it looks like a face
combo = models.Model(noise_in, isValid)
combo.compile(loss="binary_crossentropy", optimizer=opt)

#combo.summary()

# we only train one "model" at a time

def train(models, X_train, noise, result="result/", epochs=20000, batch_size=128):
    combo, disc, gen = models
    noise_latent = noise.shape[1]
    half = int(batch_size / 2)
    history = []
    
    for epoch in range(epochs):
        # start with discriminator
        
        # select half the images
        indices = np.random.randint(0, X_train.shape[0], half)
        images = X_train[indices]
        disc_noise = get_noise(half, noise_latent)
        
        # generate some new images
        gen_images = gen.predict(disc_noise)
        
        # train the discriminator!!
        disc_loss_real = disc.train_on_batch(images, np.ones((half, 1)))
        disc_loss_fake = disc.train_on_batch(gen_images, np.zeros((half, 1)))
        #print(disc_loss_real.shape, "", disc_loss_fake.shape)
        disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)
        
        # train generator
        
        gen_noise = get_noise(batch_size, noise_latent)
        
        # we want all generated images to be labeled as valid
        valid = (np.array([1]*batch_size)).reshape(batch_size, 1)
        
        combo_loss = combo.train_on_batch(gen_noise, valid)
        
        history.append({"D":disc_loss[0], "G":combo_loss})
        
        if epoch % 100 == 0:
            # Plot the progress
            print ("Epoch {:05.0f} [D loss: {:4.3f}, acc.: {:05.1f}%] [G loss: {:4.3f}]".format(
                    epoch, disc_loss[0], 100*disc_loss[1], combo_loss))
        if epoch % int(epochs/100) == 0:
            plot_generated_images(noise,
                                  path_save=result+"/image_{:05.0f}.png".format(epoch),
                                  title="Epoch {}".format(epoch))
        #if epoch % 1000 == 0:
            #plot_generated_images(noise,
            #                      title="Epoch {}".format(epoch))

        
        #if epoch % int(epochs/100) == 0:
        #    path = result + "/image_%i.png" % epoch
        #    title = "Epoch %i" % epoch
        #    plot_generated_images(noise, path_save=path, titleadd=title)
    
    return history

noise = get_noise(nsample=4, latent_dim=100)

history = train((combo, discriminator, generator), X_train, noise=noise, epochs=10000, batch_size=128)