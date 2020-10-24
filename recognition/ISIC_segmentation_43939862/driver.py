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
    lbl = tf.squeeze(lbl)
    lbl = tf.expand_dims(lbl, -1)
    lbl = tf.keras.backend.round(lbl / 255.0)
    lbl.set_shape([256,256,1])
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
test_label = np.asarray(label_batch[0][:,:,0])
plt.figure(figsize = (20,20))
plt.subplot(1,2,1); plt.imshow(test_image); plt.title('Image'); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(test_label); plt.title('Label'); plt.axis('off')
plt.show()

#Build model
model = ImprovedUnet(h, w, n_channels)
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        layer.add_loss(lambda: tf.keras.regularizers.l2(0.001)(layer.kernel))
model.summary()

#Compile model
def dice(lbl_gt, lbl_pred):
    lbl_gt = tf.keras.backend.flatten(lbl_gt)
    lbl_pred = tf.keras.backend.flatten(lbl_pred)
    intersection = tf.keras.backend.sum(lbl_gt * lbl_pred)
    union = tf.keras.backend.sum(lbl_gt) + tf.keras.backend.sum(lbl_pred)
    return (2.0 * intersection + 1.0) / (union + 1.0)

def dice_loss(lbl_gt, lbl_pred):
    return 1-dice(lbl_gt, lbl_pred)

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
print('Test set evaluation:\n')
model.evaluate(test_ds.batch(16), verbose=2)

preds = model.predict(test_ds.batch(16))
labels = [lb for im, lb in list(test_ds.as_numpy_iterator())]
dice_scores = [dice(labels[i],preds[i]).numpy() for i in range(len(preds))]
print('Test set prediction mean DICE: ', sum(dice_scores)/len(dice_scores))

#Predictions
image_batch, label_batch = next(iter(test_ds.batch(3)))
pred = model.predict(image_batch)
plt.figure(figsize = (20,20))

for i in range(3):
    pred_im = tf.cast(image_batch[i], tf.float32) + tf.image.grayscale_to_rgb(tf.cast(pred[i], tf.float32))
    gt_im = tf.cast(image_batch[i], tf.float32) + tf.image.grayscale_to_rgb(tf.cast(label_batch[i], tf.float32))
    plt.subplot(3,2,2*i+1); plt.imshow(gt_im); plt.title('Ground truth'); plt.axis('off')
    plt.subplot(3,2,2*i+2); plt.imshow(pred_im); plt.title('Prediction'); plt.axis('off')
plt.show()