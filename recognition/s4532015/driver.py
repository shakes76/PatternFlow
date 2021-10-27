#where the paramters are defined and models run
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from models import make_style_generator, make_synthesis_network, make_generator_model, make_discriminator_model
from functions import train, make_gif


#parameters
batch_size = 8
depth = 8          #filters
latent_size = 64   #size of input vector z
im_size = 256       #final image size
n_layers = 8        #no. layers in the synthesis network
epochs = 5         #no. of epochs to run the training for


#load data here
data_path = "D:\Datasets\keras_png_slices_data\keras_png_slices_train"
raw_ds = tf.keras.preprocessing.image_dataset_from_directory(data_path, labels=None, color_mode='grayscale', batch_size=batch_size)

#normalise the data
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
norm_ds = raw_ds.map(lambda img: (normalization_layer(img)))

#shuffle the data
norm_ds = norm_ds.shuffle(norm_ds.cardinality().numpy())


#define models
G = make_synthesis_network(n_layers, im_size, batch_size, depth)
S = make_style_generator(latent_size)
gen_model = make_generator_model(S, G, n_layers, latent_size, im_size)
D = make_discriminator_model(im_size, depth)

#optimisers
gen_model_optimiser = Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.999)
D_optimiser = Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.999)


#train the models
disc_loss, gen_loss = train(S, G, D, gen_model, gen_model_optimiser, D_optimiser, batch_size, im_size, n_layers, latent_size, norm_ds, epochs)

#graph the losses
x = np.arange(0, epochs)
plt.plot(x, disc_loss) #blue    >discriminator loss
plt.plot(x, gen_loss) #orange   >generator loss
plt.show()

#make a gif of the training
make_gif()