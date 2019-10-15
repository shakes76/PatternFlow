## load modules
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Input, UpSampling2D
from skimage import color, data, restoration
from scipy.signal import convolve2d
# In[2]:

astro = color.rgb2gray(data.astronaut())

# In[3]:

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                       sharex=True, sharey=True)

plt.gray()

ax[0].axis('off')
ax[0].set_title('Data')

ax[1].imshow(astro)
ax[1].axis('off')
ax[1].set_title('Self tuned restoration')

fig.tight_layout()

plt.show()
