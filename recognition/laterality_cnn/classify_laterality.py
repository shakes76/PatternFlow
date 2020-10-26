import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import random
import glob

#Specify directory of data
data_dir = os.path.join("C:", "Users", "delic", ".keras", "datasets", "AKOA_Analysis")

#Split the dataset - 20% validation, 80% training
batch_size = 32
img_height = 228
img_width = 260

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels = 'inferred',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size = (img_height, img_width),
    batch_size = batch_size,
    color_mode = 'grayscale'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels = 'inferred',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size = (img_height, img_width),
    batch_size = batch_size,
    color_mode = 'grayscale'
)

class_names = train_ds.class_names
num_classes = len(class_names)

# #Visualise data
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8")[:,:,0])
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
# plt.show()

# for image_batch, labels_batch in train_ds:
#     print(image_batch.shape)
#     print(np.min(image_batch[0]), np.max(image_batch[0]))
#     print(labels_batch.shape)
#     break

#Configure dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

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
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape= (img_height, img_width, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1E-3),
    loss=loss_fn,
    metrics=['accuracy']
)

# results = model.load_weights(checkpoint_path)

results = model.fit(train_ds, epochs=n_epochs, callbacks=[cp_callback], validation_data=val_ds)

model.evaluate(val_ds, verbose=2)
model.evaluate(train_ds, verbose=2)

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
