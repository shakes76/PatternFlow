'''
Segment the OASIS brain data set with an Improved UNet.

@author Aghnia Prawira (45610240)
'''

from glob import glob
import data_processing as dp
import improved_unet as iu
import matplotlib.pyplot as plt
import random
import tensorflow as tf

import sys
print('Python version:', sys.version)
print('TensorFlow version:', tf.__version__)

# Set path to dataset
dataset_path = "keras_png_slices_data/"

seg_test_path = sorted(glob(dataset_path + "keras_png_slices_seg_test/*.png"))
seg_train_path = sorted(glob(dataset_path + "keras_png_slices_seg_train/*.png"))
seg_val_path = sorted(glob(dataset_path + "keras_png_slices_seg_validate/*.png"))
test_path = sorted(glob(dataset_path + "keras_png_slices_test/*.png"))
train_path = sorted(glob(dataset_path + "keras_png_slices_train/*.png"))
val_path = sorted(glob(dataset_path + "keras_png_slices_validate/*.png"))


# Create tensorflow dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_path, seg_train_path))
val_ds = tf.data.Dataset.from_tensor_slices((val_path, seg_val_path))
test_ds = tf.data.Dataset.from_tensor_slices((test_path, seg_test_path))


print("Loading and processing images ...")
# Load and process images
train_ds = train_ds.map(dp.process_image)
val_ds = val_ds.map(dp.process_image)
test_ds = test_ds.map(dp.process_image)

test, seg_test = next(iter(test_ds.batch(len(test_path))))
        

print("Generating model ...")
# Generate improved unet model
model = iu.unet()
model.summary()


print("Compiling model ...")
# Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', metrics=[iu.dice_coefficient_avg])


print("Training model ...")
# Train the model
history = model.fit(train_ds.batch(20), 
                    epochs=8, 
                    validation_data=val_ds.batch(20))


# Plot the training and validation DSC
plt.figure(figsize=(8, 5))
plt.title("Dice Similarity Coefficient")
plt.plot(history.history["dice_coefficient_avg"], label="Training DSC")
plt.plot(history.history["val_dice_coefficient_avg"], label="Validation DSC")
plt.xlabel("Epoch")
plt.legend();
plt.show()


# Calculate DSC
prediction = model.predict(test)
tf.print("Average DSC for all labels: ", iu.dice_coefficient_avg(seg_test, prediction))
tf.print("DSC for each label: ", iu.dice_coefficient(seg_test, prediction))


# Display random predictions
def display(title_list, image_list, cmap='viridis'):
    fig, ax = plt.subplots(1, len(title_list), figsize=(10, 10))
    for j, k in enumerate(title_list):
        ax[j].set_title(k)
    for j, k in enumerate(image_list):
        ax[j].imshow(k, cmap=cmap)
    plt.show()

random_images = [random.randint(1,len(test)) for i in range(3)]

for i in random_images:
    display(['Input Image', 'True Segmentation', 'Predicted Segmentation'], 
            [test[i][:,:,0], tf.argmax(seg_test[i], axis=-1), tf.argmax(prediction[i], axis=-1)], cmap='gray')