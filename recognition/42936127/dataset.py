import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from PIL import Image
import tensorflow_datasets as tfds

seed = 123
batch_size = 64
img_height = 256
img_width = 256



dataset = tf.keras.utils.image_dataset_from_directory(
    "keras_png_slices_data", 
    labels = "inferred",
    seed = seed,
    image_size= (img_height, img_width)
    )

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     "keras_png_slices_data", 
#     labels = "inferred",
#     validation_split = 0.2,
#     subset="validation",
#     seed = seed,
#     image_size= (img_height, img_width)
#     #batch_size = batch_size
#     )


print("training and validation loaded")

class_names = train_ds.class_names
print(class_names)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

# plt.show()
# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break

normalization_layer = tf.keras.layers.Rescaling(1./255)
#normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

#image_batch, labels_batch = next(iter(normalized_ds))
#first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
#print(np.min(first_image), np.max(first_image), labels_batch[0])


#AUTOTUNE = tf.data.AUTOTUNE

# train_ds = normalized_ds.cache().prefetch()
# val_ds = normalized_val_ds.cache().prefetch()

# train_ds = normalized_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = normalized_val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# builder = tfds.ImageFolder(os.path.join(os.getcwd(), "keras_png_slices_data"))
# print(builder.info)  # num examples, labels... are automatically calculated
# ds_train = builder.as_dataset(split="train", shuffle_files=True, as_supervised = True)
# ds_test = builder.as_dataset(split="test", shuffle_files=False, as_supervised = True)
# ds_validation = builder.as_dataset(split="validate", shuffle_files=True , as_supervised = True)

# AUTOTUNE = tf.data.experimental.AUTOTUNE
# BATCH_SIZE = 64

# ds_train = ds_train.shuffle(builder.info.splits["train"].num_examples)

# tfds.show_examples(ds_train, builder.info , rows = 3, cols=3)