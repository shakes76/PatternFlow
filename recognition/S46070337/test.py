import numpy as np
import tensorflow as tf
from model import unet_model
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import zipfile

dataset_url = "https://cloudstor.aarnet.edu.au/sender/download.php?token=b66a9288-2f00-4330-82ea-9b8711d27643&files_ids=14200406"
data_path = tf.keras.utils.get_file(origin=dataset_url, fname="ISIC2018_Task1-2_Training_Data.zip")
with zipfile.ZipFile(data_path) as zf:
    zf.extractall()

# Get the images and return the sorted list
input_images = sorted(glob.glob('ISIC2018_Task1-2_Training_Input_x2/*.jpg'))
output_masks = sorted(glob.glob('ISIC2018_Task1_Training_GroundTruth_x2/*.png'))

# data preprocessing
images = []
masks = []
for i in range(len(input_images)):
    images_temp = Image.open(input_images[i])
    masks_temp = Image.open(output_masks[i])
    # resizing images
    images_temp = np.array(images_temp.resize((256, 256), Image.ANTIALIAS))
    masks_temp = np.array(masks_temp.resize((256, 256), Image.ANTIALIAS))
    # appending into list
    images.append(images_temp)
    masks.append(masks_temp)

images = np.array(images)
masks = np.expand_dims(np.array(masks), -1)

print(images.shape)
print(masks.shape)
masks = masks / 255.0
masks = np.around(masks)

X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.3, random_state=1)
X_train, val_X_train, y_train, val_y_train = train_test_split(X_train, y_train, test_size=0.15, random_state=1)

# Build the model and evaluate it
model = unet_model()
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), metrics=['accuracy'])

# model training
history = model.fit(X_train, y_train, batch_size=8, epochs=50, validation_data=(val_X_train, val_y_train))

# Prediction Results
y_pred = model.predict(X_test)

print(y_pred.shape)
print(y_test.shape)

# Model evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def dice_coefficient(y_true, y_pred, smooth=1.):
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1, 2])
    union = tf.keras.backend.sum(y_true + y_pred, axis=[1, 2])
    dice_score = tf.reduce_mean((2.0 * intersection) / (union + smooth), axis=0)
    return dice_score


dice = dice_coefficient(y_test, y_pred, smooth=1.)
print(dice)

# Out put prediction results
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
ax[0, 0].imshow(X_test[12])
ax[0, 1].imshow(y_test[12])
ax[0, 2].imshow(np.round(y_pred[12]))
plt.show()

# Evaluation curve
fig, axs = plt.subplots(1, 2, figsize=(14, 4))
axs[0].set_title("Loss plot")
axs[0].plot(history.history['loss'], color='purple', label='train')
axs[0].plot(history.history['val_loss'], color='orange', label='test')
axs[1].set_title("Accuracy plot")
axs[1].plot(history.history['accuracy'], color='purple', label='train')
axs[1].plot(history.history['val_accuracy'], color='orange', label='test')
plt.show()
