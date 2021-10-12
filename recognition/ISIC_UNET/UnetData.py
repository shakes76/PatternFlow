import tensorflow as tf
import os
import glob
import sklearn.utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot



def load_image(file, shape):
    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img, channels=3)  # so it is in colour
    img = tf.image.resize(img, shape)
    img = tf.cast(img, tf.float32) / 255.0  # need to divide
    return img


def load_seg(file, shape):
    seg = tf.io.read_file(file)
    seg = tf.image.decode_png(seg, channels=1)  # gray scale
    seg = tf.image.resize(seg, shape)
    seg = tf.cast(seg, tf.float32)  # need to divide
    seg = tf.math.round(seg / 255.0)  # need to put in binary
    binary = (seg == [0.0, 1.0])  # @@@ dont think i need it
    binary_seg = tf.cast(binary, tf.float32)
    return binary_seg


def process_images(image_path, mask_path):
    size = (256, 256)
    image = load_image(image_path, size)
    segmentation = load_seg(mask_path, size)

    return image, segmentation


# https://matplotlib.org/stable/tutorials/introductory/images.html
def view_images(data, num):
    matplotlib.pyplot.figure(figsize=(4, num*4))
    loc = 0
    for img, label in data.take(num):
        matplotlib.pyplot.subplot(num, 2, 2*loc+1)
        matplotlib.pyplot.imshow(img)
        matplotlib.pyplot.subplot(num, 2, 2*loc+2)
        matplotlib.pyplot.imshow(label[:,:,1], cmap='gray')
        loc = loc + 1
    matplotlib.pyplot.show()


def prepare_images():  # x_path, mask_path
    x_images = 'C:\\Users\\Mark\\Downloads\\ISIC2018_Task1-2_Training_Data\\ISIC2018_Task1-2_Training_Input_x2\\*.jpg'
    mask_images = 'C:\\Users\\Mark\\Downloads\\ISIC2018_Task1-2_Training_Data\\ISIC2018_Task1_Training_GroundTruth_x2\\*.png'
    x_images = sorted(glob.glob(x_images))
    mask_images = sorted(glob.glob(mask_images))

    print(len(x_images))
    print(len(mask_images))

    X_train, X_test, y_train, y_test = train_test_split(x_images, mask_images, test_size=0.2, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    final_train = train_ds.shuffle(len(X_train))
    final_test = test_ds.shuffle(len(X_test))
    final_val = val_ds.shuffle(len(X_val))



    final_train = final_train.map(process_images)
    final_test = final_test.map(process_images)
    final_val = final_val.map(process_images)

    view_images(final_train, 3)
    return final_train, final_test, final_val

