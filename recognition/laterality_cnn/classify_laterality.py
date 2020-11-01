import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import random
import glob

#Specify directory of data
data_dir = os.path.join("C:", "Users", "delic", ".keras", "datasets", "AKOA_Analysis")

#Load all the filenames
filenames = glob.glob(os.path.join(data_dir, '*', '*.png'))
image_count = len(filenames)

#Split filenames by patiend id
patient_files = dict()
for fn in filenames:
    pid = fn.split(os.path.sep)[-1].split('_')[0]
    if pid not in patient_files:
        patient_files[pid] = [fn]
    else:
        patient_files[pid].append(fn)

#Count number of patients
patient_ids = list(patient_files.keys())
patient_count = len(patient_ids)


#Split datasets (validation, test, training)
random.seed(123)
not_valid = True
while not_valid:
    random.shuffle(patient_ids)

    #Split the patients - 16% validation, 20% test, 64% training
    val_count = int(patient_count * 0.16)
    test_count = int(patient_count * 0.36)
    val_patients = patient_ids[:val_count]
    test_patients = patient_ids[val_count:test_count]
    train_patients = patient_ids[test_count:]

    val_images = []
    test_images = []
    train_images = []

    #Add every patient's filenames to their respective image datasets
    for pid in val_patients:
        val_images.extend(patient_files[pid])
    for pid in test_patients:
        test_images.extend(patient_files[pid])
    for pid in train_patients:
        train_images.extend(patient_files[pid])

    # print(len(val_patients), len(test_patients), len(train_patients))
    # print(len(val_images), len(test_images), len(train_images))

    #Extract labels
    train_labels = [fn.split(os.path.sep)[-2] for fn in train_images]
    val_labels = [fn.split(os.path.sep)[-2] for fn in val_images]
    test_labels = [fn.split(os.path.sep)[-2] for fn in test_images]
    not_valid = False #Break while loop

    #Count number of images in each dataset labelled 'right'
    train_right_count = 0
    for label in train_labels:
        if label == 'right':
            train_right_count += 1

    val_right_count = 0
    for label in val_labels:
        if label == 'right':
            val_right_count += 1

    test_right_count = 0
    for label in test_labels:
        if label == 'right':
            test_right_count += 1

    #Ratios of 'right' labelled to 'left' labelled
    right_ratios = [train_right_count/len(train_labels), val_right_count/len(val_labels), test_right_count/len(test_labels)]
    # print(right_ratios)

    #If too many or too few 'right' images, re-shuffle and re-split datasets.
    for ratio in right_ratios:
        if ratio > 0.7 or ratio < 0.3:
            print("Re-shuffling")
            not_valid = True #continue while loop
  
#Class names list and number of classes
class_names = sorted(set(train_labels))
num_classes = len(class_names)

#Create tensorflow datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

#Map filenames and labels to data arrays
img_height = 228
img_width = 260
def map_fn(filename, label):
    # Load the raw data from the file as a string.
    img = tf.io.read_file(filename)
    # Convert the compressed string to a 3D uint8 tensor.
    img = tf.image.decode_jpeg(img, channels=1) # channels=3 for RGB, channels=1 for grayscale
    # Resize the image to the desired size.
    img = tf.image.resize(img, (img_height, img_width))
    # Standardise values to be in the [0, 1] range.
    img = tf.cast(img, tf.float32) / 255.0
    # One-hot encode the label.
    one_hot = tf.cast(label == class_names, tf.uint8)
    # Return the processed image and label.
    return img, one_hot

#Apply mapping to datasets
train_ds = train_ds.map(map_fn)
val_ds = val_ds.map(map_fn)
test_ds = test_ds.map(map_fn)

# #Visualise data
# val_ds = val_ds.shuffle(len(val_images))
# image_batch, label_batch = next(iter(val_ds.batch(9)))

# plt.figure(figsize=(10, 10))
# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(image_batch[i][:,:,0])
#     label = tf.argmax(label_batch[i])
#     plt.title(class_names[label])
#     plt.axis('off')
# plt.show()

#Configure dataset for performance and shuffling. Shuffle buffer = number of images
batch_size = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(len(train_images))
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache()
val_ds = val_ds.shuffle(len(val_images))
val_ds = val_ds.batch(batch_size)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache()
test_ds = test_ds.shuffle(len(test_images))
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

#Create ModelCheckpoint to save and load trained models
checkpoint_path = "training/ckpt01.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
    save_freq='epoch'
)

#Create subclasses for layers and model
class ConvBlock(tf.keras.layers.Layer):
    """
    Subclass for a convolution block (2 convolution layers, 1 maxpooling layer and 1 dropout layer)
    """
    def __init__(self, filters=32):
        super(ConvBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2))
        self.drop = tf.keras.layers.Dropout(0.2)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        return self.drop(x)

class CNNModel(tf.keras.Model):
    """
    Subclass for the CNN model. 3 convolution blocks, 1 flatten layer, 2 dense layers and 1 dropout layer.
    """
    def __init__(self, num_classes=2):
        super(CNNModel, self).__init__()
        self.block1 = ConvBlock(32)
        self.block2 = ConvBlock(64)
        self.block3 = ConvBlock(128)
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.5)
        self.d2sm = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.drop(x)
        return self.d2sm(x)

#Create an instance of the model
model = CNNModel(num_classes=num_classes)

#Use the model to predict an image to automatically 'build' it and allow calling of model.summary()
image_batch, label_batch = next(iter(val_ds.take(1)))
predictions = model.predict(image_batch)

#Check model summary
model.summary()

#Loss function
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

#Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1E-3),
    loss=loss_fn,
    metrics=['accuracy']
)

#Load saved weights
results = model.load_weights(checkpoint_path)

#Train the model
n_epochs = 10
# results = model.fit(train_ds, epochs=n_epochs, callbacks=[cp_callback], validation_data=val_ds)

#Check final weights by evaluating on all sets
print("Training set: ")
model.evaluate(train_ds, verbose=2)
print("Validation set: ")
model.evaluate(val_ds, verbose=2)
print("Test set: ")
model.evaluate(test_ds, verbose=2)

#Plot accuracy vs validation accuracy and loss vs validation loss (for every epoch)
# plt.plot(results.history['accuracy'], label='accuracy')
# plt.plot(results.history['val_accuracy'], label='val_accuracy')
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.ylim(0.5, 1.0)
# plt.legend()
# plt.savefig('accuracy.png')
# plt.show()

# plt.plot(results.history['loss'], label="loss")
# plt.plot(results.history['val_loss'], label="val_loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.ylim(0.0, 0.5)
# plt.legend()
# plt.savefig('loss.png')
# plt.show()
