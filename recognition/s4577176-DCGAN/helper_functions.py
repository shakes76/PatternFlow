from libraries import *

#generator:

def generator(img_shape, noise_shape = (100,)):
    '''
    noise_shape : the dimension of the input vector for the generator
    img_shape   : the dimension of the output
    '''

    input_noise = layers.Input(shape=noise_shape) 
    d = layers.Dense(1024, activation="relu")(input_noise) 
    d = layers.Dense(1024, activation="relu")(input_noise) 
    d = layers.Dense(128*8*8, activation="relu")(d)
    d = layers.Reshape((8,8,128))(d)
    
    d = layers.Conv2DTranspose(128, kernel_size=(2,2),strides=(2,2), use_bias=False)(d)
    d = layers.Conv2D(64 ,(1,1),activation='relu',padding='same')(d)


    d = layers.Conv2DTranspose(32,kernel_size=(2,2),strides=(2,2),use_bias=False)(d)
    d = layers.Conv2D(64,(1,1),activation='relu',padding='same')(d) 
    
    if img_shape == 64:
        d = layers.Conv2DTranspose(32,kernel_size=(2,2),strides=(2,2),use_bias=False)(d)
        d = layers.Conv2D(64,(1,1),activation='relu',padding='same')(d) 
        
    if img_shape == 128:
        d = layers.Conv2DTranspose(32,kernel_size=(2,2),strides=(2,2),use_bias=False)(d)
        d = layers.Conv2D(64,(1,1),activation='relu',padding='same')(d)
        
        d = layers.Conv2DTranspose(32,kernel_size=(2,2),strides=(2,2),use_bias=False)(d)
        d = layers.Conv2D(64,(1,1),activation='relu',padding='same')(d)
        
    if img_shape == 256:
        d = layers.Conv2DTranspose(32,kernel_size=(2,2),strides=(2,2),use_bias=False)(d)
        d = layers.Conv2D(64,(1,1),activation='relu',padding='same')(d)
        
        d = layers.Conv2DTranspose(32,kernel_size=(2,2),strides=(2,2),use_bias=False)(d)
        d = layers.Conv2D(64,(1,1),activation='relu',padding='same')(d)
        
        d = layers.Conv2DTranspose(32,kernel_size=(2,2),strides=(2,2),use_bias=False)(d)
        d = layers.Conv2D(64,(1,1),activation='relu',padding='same')(d)
    
    
    img = layers.Conv2D(3,(1,1),activation='sigmoid',padding='same')(d)
    
    model = models.Model(input_noise, img)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.00007, 0.5))
    
    return model


#discriminator:

def build_discriminator(img_shape):

    input_img = layers.Input(shape=img_shape)
    
    x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(input_img)
    x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
    x = layers.MaxPooling2D((2,2),strides=(2, 2))(x)
    
    x = layers.Conv2D(64,(3, 3),activation='relu',padding='same')(x)
    x = layers.Conv2D(64,(3, 3),activation='relu',padding='same')(x)
    x = layers.MaxPooling2D((2, 2),strides=(2, 2))(x)
    
    x = layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x = layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x = layers.MaxPooling2D((2,2),strides=(1,1))(x)

    
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    out = layers.Dense(1,activation='sigmoid')(x)
    
    model     = models.Model(input_img, out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.00007, 0.5),metrics=['accuracy'])

    return model


#combiner:

def combiner(noise_shape,generator,discriminator):
    z = layers.Input(shape=noise_shape)
    img = generator(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The valid takes generated images as input and determines validity
    valid = discriminator(img)

    # The combined model  (stacked generator and discriminator) takes
    # noise as input => generates images => determines validity 
    combined = models.Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.00007, 0.5))
    return combined

#noise generator
def get_noise(nsample=1, nlatent_dim=100):
    noise = np.random.normal(0, 1, (nsample,nlatent_dim))
    return(noise)

#random image plotter:
def plot_generated_images(noise,generator,nsample,path_save=None,titleadd=""):
    imgs = generator.predict(noise)
    fig = plt.figure(figsize=(40,10))
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(1,nsample,i+1)
        ax.imshow(img)
    fig.suptitle("Generated images "+titleadd,fontsize=30)
    
    if path_save is not None:
        plt.savefig(path_save,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()
    else:
        plt.show()
