import glob
import tensorflow as tf
from model import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard

#Load data
image_files = glob.glob('../../../ISIC2018_Data/ISIC2018_Images/*.jpg')
label_files = glob.glob('../../../ISIC2018_Data/ISIC2018_Labels/*.png')

split_ratio = 0.8
split_size = int(len(image_files)*split_ratio)
data_split_size = int(len(image_files)*split_ratio*split_ratio)

#Shuffle before splitting?
train_ds = tf.data.Dataset.from_tensor_slices((image_files[:data_split_size], 
                                               label_files[:data_split_size]))
validate_ds = tf.data.Dataset.from_tensor_slices((image_files[data_split_size:split_size], 
                                               label_files[data_split_size:split_size]))
test_ds = tf.data.Dataset.from_tensor_slices((image_files[split_size:], 
                                               label_files[split_size:]))

#Pre-process data
train_ds = train_ds.shuffle(data_split_size)
validate_ds = validate_ds.shuffle(split_size-data_split_size)
test_ds = test_ds.shuffle(len(image_files)-split_size)

def map_fn(image, label):
    img = tf.io.read_file(image)
    img = tf.io.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, (256,256))
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([256,256,3])
    
    lbl = tf.io.read_file(label)
    lbl = tf.io.decode_png(lbl, channels = 0)
    lbl = tf.image.resize(lbl, (256,256))
    lbl = tf.keras.backend.round(lbl / 255.0) #Is this fine?
    lbl = tf.cast(lbl, tf.uint8)
    lbl = tf.one_hot(lbl, depth = 2, axis = 2) #Is one-hot encoding needed for only 2 classes?
    lbl = tf.cast(lbl, tf.float32)
    lbl = tf.squeeze(lbl)
    lbl.set_shape([256,256,2])
    return img, lbl

train_ds = train_ds.map(map_fn)
validate_ds = validate_ds.map(map_fn)
test_ds = test_ds.map(map_fn)

#Check batching is working
for image, label in train_ds.take(1):
    print('Image shape: ', image.numpy().shape)
    print('Label shape: ', label.numpy().shape)
    h, w, n_channels = image.numpy().shape

#Print data size info
print("\nData info:")
print("height: %d" % h)
print("width: %d" % w)
print("channels: %d" % n_channels)

image_batch, label_batch = next(iter(train_ds.batch(1)))
test_image = np.asarray(image_batch[0])
test_label = np.asarray(label_batch[0][:,:,1])
plt.figure(figsize = (20,20))
plt.subplot(1,2,1); plt.imshow(test_image); plt.title('Image'); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(test_label); plt.title('Label'); plt.axis('off')
plt.show()

#Build model
input_layer, output_layer = ImprovedUnet(h, w, n_channels)
model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
model.summary()

#Compile model
def dice(lbl_gt, lbl_pred):
    lbl_gt = tf.keras.backend.flatten(lbl_gt)
    lbl_pred = tf.keras.backend.flatten(lbl_pred)
    intersection = tf.keras.backend.sum(lbl_gt * lbl_pred)
    return (2.0 * intersection + 1) / (tf.keras.backend.sum(lbl_gt) + tf.keras.backend.sum(lbl_pred) + 1)

def dice_loss(lbl_gt, lbl_pred):
    return -dice(lbl_gt, lbl_pred)

model.compile(optimizer = 'Adam', loss=dice_loss, metrics=[dice]) 

#Train model
history = model.fit(train_ds.batch(16), epochs=100, validation_data = validate_ds.batch(16), 
                callbacks = [TensorBoard(log_dir='./tb', histogram_freq=0, write_graph=False, profile_batch = 100000000)])

#Evaluate model
plt.figure(1)
plt.plot(history.history['dice'])
plt.plot(history.history['val_dice'])
plt.title('Model Accuracy (DICE) vs Epoch')
plt.ylabel('Accuracy (DICE)')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='lower right')
plt.show()
print('Test set summary:\n')
model.evaluate(test_ds.batch(16), verbose=2)

#Predictions
image_batch, label_batch = next(iter(test_ds.batch(3)))
pred = model.predict(image_batch)
plt.figure(figsize = (20,20))

for i in range(3):
    pred_im = tf.cast(image_batch[i], tf.float32) + tf.image.grayscale_to_rgb(tf.expand_dims(tf.cast(tf.argmax(pred[i], axis = 2), tf.float32), 2))
    gt_im = tf.cast(image_batch[i], tf.float32) + tf.image.grayscale_to_rgb(tf.expand_dims(tf.cast(tf.argmax(label_batch[i], axis = 2), tf.float32), 2))
    plt.subplot(3,2,2*i+1); plt.imshow(gt_im); plt.title('Ground truth'); plt.axis('off')
    plt.subplot(3,2,2*i+2); plt.imshow(pred_im); plt.title('Prediction'); plt.axis('off')
plt.show()