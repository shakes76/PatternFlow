import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt

def map_fn(image_filename, label_filename):
    # if filename.endswith('.jpg'):
    # # input image
    img = tf.io.read_file(image_filename)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    img = img / 255.
    # img = img[np.newaxis, :, :, :] # TODO

    # ground truth image
    label = tf.io.read_file(label_filename)
    label = tf.io.decode_jpeg(label, channels=1)
    label = tf.image.resize(label, (256, 256))
    label = label / 255.
    label = tf.cast(label > 0.5, dtype=tf.float32)  # mask

    return img, label


dataset_dir = 'ISIC2018_Task1-2_Training_Data/'
input_dir = dataset_dir + 'ISIC2018_Task1-2_Training_Input_x2/'
groundtruth_dir = dataset_dir + 'ISIC2018_Task1_Training_GroundTruth_x2/'

input_filenames = sorted(glob.glob(input_dir + '/*.jpg'))
groundtruth_filenames = sorted(glob.glob(groundtruth_dir + '/*.png'))

# sanity check
assert len(input_filenames) == len(groundtruth_filenames)
test_size = int(len(input_filenames) * .2)

# train, val, test split
train_images = input_filenames[2*test_size:]
train_labels = groundtruth_filenames[2*test_size:]

val_images = input_filenames[test_size:2*test_size]
val_labels = groundtruth_filenames[test_size:2*test_size]

test_images = input_filenames[:test_size]
test_labels = groundtruth_filenames[:test_size]

# combine input images and ground truth images (file names)
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# load images
train_ds = train_ds.map(map_fn)
val_ds = val_ds.map(map_fn)
test_ds = test_ds.map(map_fn)


# visualization
input_batch, groundtruth_batch = next(iter(test_ds.batch(3)))

for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.imshow(input_batch[i].numpy())
    plt.title('image ' + str(i+1))
    plt.axis('off')

    plt.subplot(2, 3, i+4)
    plt.imshow(groundtruth_batch[i].numpy(), cmap='gray')
    plt.title('ground truth ' + str(i+1))
    plt.axis('off')

plt.show()
