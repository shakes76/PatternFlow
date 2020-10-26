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

#Split the dataset - 20% validation, 20% test, 60% training
random.seed(123)
random.shuffle(filenames)
val_size = int(image_count * 0.2)
test_size = int(image_count * 0.4)
val_images = filenames[:val_size]
test_images = filenames[val_size:test_size]
train_images = filenames[test_size:]

#Extract labels
train_labels = [fn.split(os.path.sep)[-2] for fn in train_images]
val_labels = [fn.split(os.path.sep)[-2] for fn in val_images]
test_labels = [fn.split(os.path.sep)[-2] for fn in test_images]

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

train_ds = train_ds.map(map_fn)
val_ds = val_ds.map(map_fn)
test_ds = test_ds.map(map_fn)

# #Visualise data
# image_batch, label_batch = next(iter(train_ds.batch(9)))

# plt.figure(figsize=(10, 10))
# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(image_batch[i][:,:,0])
#     label = tf.argmax(label_batch[i])
#     plt.title(class_names[label])
#     plt.axis('off')
# plt.show()

#Configure dataset for performance and shuffle
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

checkpoint_path = "training/ckpt01.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

n_epochs = 5

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
    save_freq='epoch'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1E-3),
    loss=loss_fn,
    metrics=['accuracy']
)

# results = model.load_weights(checkpoint_path)

results = model.fit(train_ds, epochs=n_epochs, callbacks=[cp_callback], validation_data=val_ds)

print("Training set: ")
model.evaluate(train_ds, verbose=2)
print("Validation set: ")
model.evaluate(val_ds, verbose=2)
print("Test set: ")
model.evaluate(test_ds, verbose=2)

plt.plot(results.history['accuracy'], label='accuracy')
plt.plot(results.history['val_accuracy'], label='val_accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0.5, 1.0)
plt.legend()
plt.savefig('accuracy.png')
plt.show()

plt.plot(results.history['loss'], label="loss")
plt.plot(results.history['val_loss'], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0.0, 0.5)
plt.legend()
plt.savefig('loss.png')
plt.show()
