import os
import tensorflow as tf
from tensorflow.keras import layers, models
import glob
from Model import improved_model
img_input = "C:/Users/s4588299/Downloads/ISIC2018_Task1-2_Training_Input_x2"
img_output = "C:/Users/s4588299/Downloads/ISIC2018_Task1_Training_GroundTruth_x2"

"""
Decodes a jpg into a tensor and resizes it
Parameters:
    path: The file path of the image
Returns:
    Tensor of the image
"""

def decode_jpg(path):
    jpg = tf.io.read_file(path)
    jpg = tf.image.decode_jpeg(jpg, channels = 1)
    jpg = tf.image.resize(jpg, (256, 256))
    return jpg

"""
Decodes a png into a tensor and resizes it
Parameters:
    path: The file path of the image
Returns:
    Tensor of the image
"""
def decode_png(path):
    png = tf.io.read_file(path)
    png = tf.image.decode_png(png, channels = 1)
    png = tf.image.resize(png, (256, 256))
    return png

"""
Process the label and scan to a tensor and normalise them
Parameters:
    scan_path: the file path of the scan
    label_path: the file path of the label
Returns:
    A tuple of the scan and the label
"""
def process_path(scan_path, label_path):
    #Process the scan
    scan = decode_jpg(scan_path)
    scan = tf.cast(scan, tf.float32) / 255.0
    
    label = decode_png(label_path)
    label = tf.cast(label, tf.float32) / 255.0
    label = tf.math.round(label)
    return scan, label
	
"""
Plots the loss at each epoch for the training data vs the validation data
Parameters:
	history: The data collected from the model
"""
def plot_loss(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Test vs Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("loss.png")
    plt.show()

"""
Plots the accuracy at each epoch for the training data vs the validation data
Parameters:
	history: The data collected from the model
"""
def plot_acc(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Test vs Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig("accuracy.png")
    plt.show()

#Load data
inputs = sorted(glob.glob(img_input + "/*.jpg"))
outputs = sorted(glob.glob(img_output + "/*.png"))

#Size of training, validation and test datasets
train_size = 1816
val_size = 389
test_size = 389

#Split into respective sets
scans_train = inputs[:train_size]
labels_train = outputs[:train_size]
scans_val = inputs[train_size:train_size + val_size]
labels_val = outputs[train_size:train_size + val_size]
scans_test = inputs[train_size + val_size:]
labels_test = outputs[train_size + val_size:]

train_ds = tf.data.Dataset.from_tensor_slices((scans_train, labels_train))
val_ds = tf.data.Dataset.from_tensor_slices((scans_val, labels_val))
test_ds = tf.data.Dataset.from_tensor_slices((scans_test, labels_test))

train_ds = train_ds.map(process_path)
val_ds = val_ds.map(process_path)
test_ds = test_ds.map(process_path)

def main():
    
    model = improved_model()
    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    #print(model.summary())
    history = model.fit(train_ds.batch(16), epochs = 15, validation_data = val_ds.batch(16))
    plot_loss(history)
    plot_acc(history)
main()