#required libraries are uploaded
from libraries import *
from helper_functions import *


dir_data = "C:\\Users\\61484\\Desktop\\comp3710-demo3\\particles.js-master\\keras_png_slices_data\\keras_png_slices_train"

Ntrain        = len(os.listdir(dir_data))
nm_imgs       = np.sort(os.listdir(dir_data))
## name of the jpg files for training set
nm_imgs_train = nm_imgs[:Ntrain]
img_shape     = (64,64, 3)
nsample=4

#load,resize and scale the images
def get_npdata(nm_imgs_train):
    X_train = []
    for i, myid in tqdm(enumerate(nm_imgs_train)):
        image = load_img(dir_data + "/" + myid,
                         target_size=img_shape[:2])
        image = img_to_array(image)/255.0
        X_train.append(image)
    X_train = np.array(X_train)
    return(X_train)

X_train = get_npdata(nm_imgs_train)
print("X_train.shape = {}".format(X_train.shape))


noise_shape = (100,)
nsample = 4

#calling the generator instance
generator = generator(img_shape[0], noise_shape = noise_shape)
#calling the diceriminator instance
discriminator  = build_discriminator(img_shape)
#calling the combiner instance
combined = combiner(noise_shape,generator,discriminator)

#noise instance
noise = get_noise(nsample=nsample, nlatent_dim=noise_shape[0])


#Core function of GAN.
#
def train(models, X_train, noise_plot, epochs=10, batch_size=128):

        combined, discriminator, generator = models
        nlatent_dim = noise_plot.shape[1]
        half_batch  = int(batch_size / 2)
        history = []
        for epoch in range(epochs):

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            noise = get_noise(half_batch, nlatent_dim)

            # Generate a half batch of new images
            gen_imgs = generator.predict(noise)

            #concatenate the original half and generated half of images
            X= np.concatenate([imgs, gen_imgs])
            
            #generate the labels for the batch
            y_dis = np.concatenate([np.ones((half_batch, 1)),np.zeros((half_batch,1))])
            
            #train the discriminator on the combined batch of images.
            d_loss = discriminator.train_on_batch(X, y_dis)

            noise = get_noise(batch_size, nlatent_dim)

            valid_y = (np.array([1] * batch_size)).reshape(batch_size,1)

            #train thecombiner on the combined batch of images(freeezing the layers of discrimiantor)
            g_loss = combined.train_on_batch(noise, valid_y)

            #appending the loss
            history.append({"D":d_loss[0],"G":g_loss})
            
            if epoch % 10 == 0:
                # Plot the progress
                print ("Epoch {:05.0f} [D loss: {:4.3f}, acc.: {:05.1f}%] [G loss: {:4.3f}]".format(
                    epoch, d_loss[0], 100*d_loss[1], g_loss))
                
            if epoch % 500 == 0:
                plot_generated_images(noise_plot,generator,nsample,
                                      titleadd="Epoch {}".format(epoch))
                        
        return(history)

    
start_time = time.time()

_models = combined, discriminator, generator          

#training the model
history = train(_models, X_train, noise,epochs=10000, batch_size=256)
end_time = time.time()
print("-"*10)
print("Time took: {:4.2f} min".format((end_time - start_time)/60))

#saving models:


generator.save("C:\\Users\\61484\\comp3710_final\\PatternFlow\\recognition\\s4577176-DCGAN\\saved_models\\generator" + "-" + str(img_shape[0]) + ".h5")
#discriminator.save("C:\\Users\\61484\\comp3710_final\\PatternFlow\\recognition\\s4577176-DCGAN\\saved_models\\discriminator"+ "-" + str(img_shape[0]) + ".h5")


#plotting loss values of generator and discrimiantor:

#discriminator loss
disc_loss = [i["D"] for i in history]
#generator loss
gen_loss = [i["G"] for i in history]

plt.plot(disc_loss)
plt.plot(gen_loss)