import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import dataset
import modules
import matplotlib.pyplot as plt
# images are of 256 x 240 size

train, test = dataset.load_dataset()

print(train.shape)

plt.imshow(train[1])
plt.show()

