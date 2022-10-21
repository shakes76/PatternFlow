__author__ = 'Sam Bethune - s4353631'
__date__ = '07/11/20'
__version__ = '1.1.0'

# CONFIGURE DEPENDENCIES

import tensorflow as tf
from tensorflow.keras.optimizers import Adam as adam

from tensorflow.keras import backend

import glob
from itertools import product as myzip
import os    
import tempfile

os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib.pyplot as plt

from input_line import format_fn, path_2_tens
from metrics import smoothed_jaccard_distance, dice_coe
from model import My_UNet

# Define useful variables.

my_path = 'D:\s4353631\keras_png_slices_data\keras_png_slices_'
datasets = ['train', 'validate', 'test']
data_dict = {}

# PROCESS DATA

# Create the datasets for each element of the datasets list (train, validate,
# test), storing them in data_dict.

for dataset in datasets:
    # Extract the x and y image titles; we know these to match and could
    # generate them from a single glob.glob if preferred. However, this method
    # is useful as it allows us to check that no images are missing.
    x_paths = glob.glob(my_path + dataset + '/*.png')
    y_paths = glob.glob(my_path + 'seg_' + dataset + '/*.png')
    # Create a tf.data.Dataset from the image filepaths.
    data_dict[dataset] = tf.data.Dataset.from_tensor_slices((x_paths, y_paths))
    # Shuffle each dataset at each iteration to prevent data leakage, choosing
    # a large shuffling index to safeguard against only shuffling a small
    # portion.
    data_dict[dataset] = data_dict[dataset].shuffle(10**5, \
        reshuffle_each_iteration=True)
    # Apply the path_2_tens map, filling out the datasets.
    data_dict[dataset] = data_dict[dataset].map(path_2_tens)

# VISUALISE DATA

# Extract three samples from the training set using an iterator.
x_sample, y_sample = next(iter(data_dict['train'].batch(3)))

# The samples contain each image in the first dimension, the two pixel axes
# in the second and third dimensions and the channels in the fourth 
# dimension. Here we do not require the channel on the x image.
x_sample = tf.reshape(x_sample, [3, 256, 256])

fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(12, 10))
labels = ['Cerebrospinal Fluid ', 'Grey Matter ', 'White Matter ']

# For each sample we plot the x image as well as each formatted channel of the 
# y image, making use of the product function from itertools (here renamed 
# myzip in comparison with the standard zip tool).
for i, j in myzip(range(3), range(1, 4)):
    ax[0][i].imshow(x_sample[i], cmap='gray')
    ax[0][i].axis('off')
    ax[0][i].set_title('MRI Slice ' + str(i+1))
    
    ax[j][i].imshow(y_sample[i, :, :, j], cmap='gray')
    ax[j][i].axis('off')
    ax[j][i].set_title(labels[j-1] + str(i+1))
    
fig.suptitle('OASIS Samples')
plt.show()

# BUILD MODEL

input_img_shape = (256, 256, 1)
model = My_UNet(input_img_shape)

# We know we will not need to train for long, so can implement a steeply 
# decreasing learning rate schedule.
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    10**-3,
    decay_steps=8*10**2,
    decay_rate=0.9,
    staircase=True)

# The Adam optimizer provides stability not easily achievable with SGD etc.
my_opt = adam(learning_rate=lr_schedule)

model.compile(optimizer=my_opt, 
              loss=smoothed_jaccard_distance,
              metrics=[dice_coe])

model.build(input_shape=(None, *input_img_shape))

model.build_graph().summary()

# TRAIN MODEL

# We train for 5 epochs, noting that considerably smaller models (only two 
# Downshifts!) achieve similar results.
history = model.fit(data_dict['train'].batch(2), epochs=5,
                    validation_data=data_dict['validate'].batch(2), verbose=2)

# VISUALISE PERFORMANCE

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(history.history['dice_coe'], 'k', label='Train')
ax.plot(history.history['val_dice_coe'], 'r--', label = 'Validation')
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xlabel('Epoch')
ax.set_ylabel('Dice Coefficient')
ax.set_title('Performance vs Epoch')
ax.legend()

plt.show()

# Following our previous plotting example:
test_x, test_y_ground = next(iter(data_dict['validate'].batch(1)))

test_y_ground = tf.reshape(test_y_ground, [256, 256, 4])

test_y_predict = model.predict(test_x)
test_y_predict = tf.reshape(test_y_predict, [256, 256, 4])

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 9))
labels = ['Cerebrospinal Fluid ', 'Grey Matter ', 'White Matter ']

for i in range(3):
    ax[i][0].imshow(test_y_ground[:, :, i+1], cmap='gray')
    ax[i][0].axis('off')
    ax[i][0].set_title(labels[i] + 'Ground')
    
    ax[i][1].imshow(test_y_predict[:, :, i+1], cmap='gray')
    ax[i][1].axis('off')
    ax[i][1].set_title(labels[i] + 'Predicted')

fig.suptitle('Ground Truth vs Predictions')  
plt.show()

# Finally we compute the dice coefficient on the test set.
test_performance = model.evaluate(data_dict['test'].batch(2), verbose=2)

print('Test set dice coefficient of ' + str(test_performance[1]) + '.')
