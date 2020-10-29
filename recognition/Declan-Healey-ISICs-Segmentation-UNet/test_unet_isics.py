import tensorflow as tf
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from layers import *

print('TensorFlow version:', tf.__version__)

images = sorted(glob.glob("C:\\data\ISIC2018_Task1-2_Training_Data\ISIC2018_Task1-2_Training_Input_x2\*.jpg"))
masks = sorted(glob.glob("C:\\data\ISIC2018_Task1-2_Training_Data\ISIC2018_Task1_Training_GroundTruth_x2\*.png"))

X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size = 0.3, random_state = 42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

train_ds = train_ds.shuffle(len(X_train))
test_ds = test_ds.shuffle(len(X_test))
val_ds = val_ds.shuffle(len(X_val))

def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def decode_mask(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels = 1)
    img = tf.image.resize(img, [256, 256])
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=False, name=None)
    mask = tf.one_hot(img, depth=1, dtype=tf.uint8)
    return tf.squeeze(mask)

def process_path(image_path, mask_path):
    image = decode_image(image_path)
    mask = decode_mask(mask_path)
    image = tf.reshape(image, (256, 256, 3))
    mask = tf.reshape(mask, (256, 256, 1))
    return image, mask

train_ds = train_ds.map(process_path)
test_ds = test_ds.map(process_path)
val_ds = val_ds.map(process_path)

# def display(display_list):
#     plt.figure(figsize=(10,10))
#     for i in range(len(display_list)):
#         plt.subplot(1, len(display_list), i+1)
#         plt.imshow(display_list[i], cmap='gray')
#         plt.axis('off')
#     plt.show()

# for image, mask in train_ds.take(1):
#     display([tf.squeeze(image), tf.argmax(mask, axis = -1)])

model = unet()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# class DisplayCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs = None):
#         clear_output(wait = True)
#         sho

history = model.fit(train_ds.batch(3), epochs = 3, validation_data = val_ds.batch(32))