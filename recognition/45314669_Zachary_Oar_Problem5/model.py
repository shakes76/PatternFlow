from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join
import math


def get_filenames_from_dir(directory):
    return [f for f in listdir(directory) if isfile(join(directory, f))]


def encode_y(y):
    y = np.where(y < 0.5, 0, y)
    y = np.where(y > 0.5, 1, y)

    y = keras.utils.to_categorical(y, num_classes=2)
    return y


class SequenceGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batchsize):
        self.x, self.y, self.batchsize = x, y, batchsize

    def __len__(self):
        return math.ceil(len(self.x) / self.batchsize)

    def __getitem__(self, idx):
        x_names = self.x[idx * self.batchsize:(idx + 1) * self.batchsize]
        y_names = self.y[idx * self.batchsize:(idx + 1) * self.batchsize]
        
        # open x image names, resize, normalise and make a numpy array
        batch_x = np.array([np.asarray(Image.open("ISIC2018_Task1-2_Training_Input_x2/" 
                + file_name).resize((256, 192))) for file_name in x_names]) / 255.0

        # open y image names, resize, normalise, encode to one-hot and make a numpy array
        batch_y = np.array([np.asarray(Image.open("ISIC2018_Task1_Training_GroundTruth_x2/" 
                + file_name).resize((256, 192))) for file_name in y_names]) / 255.0
        batch_y = encode_y(batch_y)

        return batch_x, batch_y


x_names = get_filenames_from_dir("ISIC2018_Task1-2_Training_Input_x2")
y_names = get_filenames_from_dir("ISIC2018_Task1_Training_GroundTruth_x2")

# 15% of all the images are set aside as the test set
x_train_val, x_test, y_train_val, y_test = train_test_split(x_names, y_names, test_size=0.15, random_state=42)

# 17% of the non-test images are set aside as the validation set
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.17, random_state=42)

train_gen = SequenceGenerator(x_train, y_train, 4)
val_gen = SequenceGenerator(x_val, y_val, 4)
test_gen = SequenceGenerator(x_test, y_test, 4)

"""
# show some of the images as a sanity check
sanity_check_x, sanity_check_y = train_gen.__getitem__(0)
plt.figure(figsize=(10, 10))
for i in (0, 2):
    plt.subplot(2, 2, i + 1)
    plt.imshow(sanity_check_x[i])
    plt.axis('off')

    plt.subplot(2, 2, i + 2)
    plt.imshow(tf.argmax(sanity_check_y[i], axis=2))
    plt.axis('off')
plt.show()
del sanity_check_x
del sanity_check_y
"""

def dice_similarity(exp, pred):
    expected = keras.backend.batch_flatten(exp)
    predicted = keras.backend.batch_flatten(pred)
    predicted = keras.backend.round(predicted)

    expected_positive = keras.backend.sum(expected, axis=-1)
    predicted_positive = keras.backend.sum(predicted, axis=-1)
    true_positive = keras.backend.sum(expected * predicted, axis=-1)
    false_negative =  expected_positive - true_positive

    false_positive = predicted_positive - true_positive
    false_positive = tf.nn.relu(false_positive)

    numerator = 2 * true_positive + keras.backend.epsilon()
    denominator = 2 * true_positive + false_positive + false_negative + keras.backend.epsilon()

    return numerator / denominator

# standard unet model, as per lecture slides
def make_model():
    # the input shape after resizing is (batch_size, 192, 256, 3)
    input_layer = keras.layers.Input(shape=(192, 256, 3))

    conv1 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = keras.layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = keras.layers.MaxPooling2D((2, 2))(conv3)
    
    conv4 = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = keras.layers.MaxPooling2D((2, 2))(conv4)

    conv_mid = keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv_mid = keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(conv_mid)
    upsamp1 = keras.layers.UpSampling2D((2, 2))(conv_mid)

    upconv1 = keras.layers.concatenate([upsamp1, conv4])
    upconv1 = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(upconv1)
    upconv1 = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(upconv1)
    upsamp2 = keras.layers.UpSampling2D((2, 2))(upconv1)

    upconv2 = keras.layers.concatenate([upsamp2, conv3])
    upconv2 = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(upconv2)
    upconv2 = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(upconv2)
    upsamp3 = keras.layers.UpSampling2D((2, 2))(upconv2)

    upconv3 = keras.layers.concatenate([upsamp3, conv2])
    upconv3 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(upconv3)
    upconv3 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(upconv3)
    upsamp4 = keras.layers.UpSampling2D((2, 2))(upconv3)

    upconv4 = keras.layers.concatenate([upsamp4, conv1])
    upconv4 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(upconv4)
    upconv4 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(upconv4)

    conv_out = keras.layers.Conv2D(2, (1,1), padding="same", activation="softmax")(upconv4)

    model = tf.keras.Model(inputs=input_layer, outputs=conv_out)
    model.compile(optimizer = keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=[dice_similarity])

    return model


# train the model
model = make_model()
model.fit(train_gen, validation_data=val_gen, epochs=1)

# evaluate the model on the test set
model.evaluate(test_gen)

# show a generated image from the test set and compare with expected output
test_images_x, test_images_y = test_gen.__getitem__(0)
prediction = model.predict(test_images_x)

plt.figure(figsize=(10, 10))
plt.subplot(4, 2, 1)
plt.imshow(tf.argmax(prediction[0], axis=2))
plt.axis('off')
plt.title("Predicted output of model", size=14)

plt.subplot(4, 2, 2)
plt.imshow(tf.argmax(test_images_y[0], axis=2))
plt.axis('off')
plt.title("Expected output (y label) for the prediction", size=14)

plt.subplot(4, 2, 3)
plt.imshow(tf.argmax(prediction[1], axis=2))
plt.axis('off')

plt.subplot(4, 2, 4)
plt.imshow(tf.argmax(test_images_y[1], axis=2))
plt.axis('off')

plt.subplot(4, 2, 5)
plt.imshow(tf.argmax(prediction[2], axis=2))
plt.axis('off')

plt.subplot(4, 2, 6)
plt.imshow(tf.argmax(test_images_y[2], axis=2))
plt.axis('off')

plt.subplot(4, 2, 7)
plt.imshow(tf.argmax(prediction[3], axis=2))
plt.axis('off')

plt.subplot(4, 2, 8)
plt.imshow(tf.argmax(test_images_y[3], axis=2))
plt.axis('off')


plt.show()
