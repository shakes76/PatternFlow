#This function is used to load the weights of the model and generates the randomly generated image of size(x*x*3)

from libraries import *
from helper_functions import *

img_shape = (64,64,3)

#load the model
generator_model = load_model("C:\\Users\\61484\\comp3710_final\\PatternFlow\\recognition\\s4577176-DCGAN\\saved_models\\generator" + "-" + str(img_shape[0]) + ".h5")
#generate the noise
noise = get_noise(1)

#generate the random image of size(x*x*3)
rnd_img = generator_model.predict(noise)[0]

plt.imshow(rnd_img)

