from tensorflow import keras
from model import *

#Load the celeba data set
n_images = 3000
print('Loading data-set using randomisation of %d samples' % n_images)
data = load_data(n_samples=n_images)
print('Shape of processed data', str(data.shape))
print('Checking image value bounds', str(np.min(data[0])), str(np.max(data[0])))

print('defining critic')
critic = define_critic(data)
critic.summary()
print('Defining generator')
generator = define_generator2(data)
generator.summary()
print('Combining D&G to Wasser-Stein GAN')
gan = define_GAN(critic, generator)
gan.summary()

print('Training GAN')
train_wgan(generator, critic, gan, data)
