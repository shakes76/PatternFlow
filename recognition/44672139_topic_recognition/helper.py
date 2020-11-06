import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob

def shuffle_map_data(images, masks):
    data = tf.data.Dataset.from_tensor_slices((images, masks))
    data = data.shuffle(len(images))
    data = data.map(map_fn)
    return data


def split_data(files, masks, ratio1, ratio2):
    num_images = len(masks)

    val_test_size = int(num_images*ratio1)

    val_test_images = files[:val_test_size]
    train_images = files[val_test_size:]
    val_test_masks = masks[:val_test_size]
    train_masks = masks[val_test_size:]

    split = int(len(val_test_masks)*ratio2)
    val_masks = val_test_masks[split:]
    val_images = val_test_images[split:]
    test_masks = val_test_masks[:split]
    test_images = val_test_images[:split]
    return train_images, train_masks, val_masks, val_images, test_masks, test_images


def map_fn(image_fp, mask_fp):
    image = tf.io.read_file(image_fp)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (512, 512))
    image = tf.cast(image, tf.float32) /255.0
    
    mask = tf.io.read_file(mask_fp)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (512, 512))
    mask = mask == [0, 255]
    mask = tf.cast(mask, tf.uint8)
    return image, mask

def display(display_list):
    plt.figure(figsize = (10, 10))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')
        plt.show()
    
def dice_coef(true, pred, smooth=1):
    true1 = tf.keras.backend.flatten(true)
    pred1 = tf.keras.backend.flatten(pred)
    overlap = tf.keras.backend.sum(true1 * pred1)+smooth
    totalPixels = (tf.keras.backend.sum(true1) + tf.keras.backend.sum(pred1))+smooth
    return (2 * overlap) / totalPixels

def convolution(inputs, filters):
    c1 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
    return tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(c1)

def unet():
    inputs = tf.keras.layers.Input(shape=(512, 512, 3))
    
    c1 = convolution(inputs, 4)
    
    c2 = tf.keras.layers.MaxPooling2D()(c1)
    c2 = convolution(c2, 8)
    
    c3 = tf.keras.layers.MaxPooling2D()(c2)
    c3 = convolution(c3, 16)
    
    c4 = tf.keras.layers.MaxPooling2D()(c3)
    c4 = convolution(c4, 32)
    
    c5 = tf.keras.layers.MaxPooling2D()(c4)
    c5 = convolution(c5, 64)
    
    c6 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    c6 = tf.keras.layers.concatenate([c6, c4])
    c6 = convolution(c6, 32)
    
    c7 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
    c7 = tf.keras.layers.concatenate([c7, c3])
    c7 = convolution(c7, 16)

    c8 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c7)
    c8 = tf.keras.layers.concatenate([c8, c2])
    c8 = convolution(c8, 8)

    c9 = tf.keras.layers.Conv2DTranspose(4, (2, 2), strides=(2, 2), padding='same')(c8)
    c9 = tf.keras.layers.concatenate([c9, c1])
    c9 = convolution(c9, 4)

    outputs = tf.keras.layers.Conv2D(2, (1,1), activation='sigmoid')(c9)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def predictions(data, model, num=4):
    image_batch, mask_batch = next(iter(data.batch(num)))
    predict = model.predict(image_batch)
    plt.figure(figsize = (11, 11))
    for i in range(num):
        plt.subplot(2, num, i+1)
        plt.imshow(tf.argmax(mask_batch[i], axis=-1), cmap = 'gray')
        plt.axis('off')
    plt.figure(figsize = (11, 11))
    for i in range(num):
        plt.subplot(2, num, i+1)
        plt.imshow(tf.argmax(predict[i], axis=-1), cmap = 'gray')
        plt.axis('off')
    for i in range(num):
        print(dice_coef(tf.argmax(mask_batch[i], axis=-1), tf.argmax(predict[i], axis=-1)).numpy())
        