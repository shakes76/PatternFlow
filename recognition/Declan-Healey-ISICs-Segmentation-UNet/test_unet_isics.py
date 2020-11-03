import tensorflow as tf
import tensorflow.keras.backend as K
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from layers import *
from IPython.display import clear_output

print('TensorFlow version:', tf.__version__)

images = sorted(glob.glob("C:\\data\ISIC2018_Task1-2_Training_Data\ISIC2018_Task1-2_Training_Input_x2\*.jpg"))
masks = sorted(glob.glob("C:\\data\ISIC2018_Task1-2_Training_Data\ISIC2018_Task1_Training_GroundTruth_x2\*.png"))

X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size = 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

train_ds = train_ds.shuffle(len(X_train))
test_ds = test_ds.shuffle(len(X_test))
val_ds = val_ds.shuffle(len(X_val))

def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels = 1)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def decode_mask(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels = 1)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.math.round(img)
    # img = tf.cast(img, tf.uint8)
    return img

def process_path(image_path, mask_path):
    print(image_path)
    print(mask_path)
    image = decode_image(image_path)
    mask = decode_mask(mask_path)
    image = tf.reshape(image, (256, 256, 1))
    mask = tf.reshape(mask, (256, 256, 1))
    return image, mask

train_ds = train_ds.map(process_path)
test_ds = test_ds.map(process_path)
val_ds = val_ds.map(process_path)

def display(display_list):
    plt.figure(figsize=(10,10))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')
    plt.show()

# for image, mask in train_ds.take(1):
#     display([tf.squeeze(image), tf.squeeze(mask)])

def dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


# def dice_distance(y_true, y_pred):
#     print(1.0 - dice_coefficient(y_true, y_pred))
#     return 1.0 - dice_coefficient(y_true, y_pred)

model = unet()
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [dice_coefficient])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', dice_coefficient])

def show_predictions(ds, num = 1):
    for image, mask in ds.take(num):
        # pred_mask = model.predict(image)
        pred_mask = model.predict(image[tf.newaxis, ...])[0]
        print(tf.squeeze(mask))
        print(tf.squeeze(pred_mask))
        display([tf.squeeze(image), tf.squeeze(mask), tf.squeeze(pred_mask)])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        clear_output(wait=True)
        show_predictions(val_ds)

# history = model.fit(train_ds.batch(8), epochs = 30, validation_data = val_ds.batch(8))
history = model.fit(train_ds.batch(8), epochs = 10, validation_data = val_ds.batch(8), callbacks = [DisplayCallback()])