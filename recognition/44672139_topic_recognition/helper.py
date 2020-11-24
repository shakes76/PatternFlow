import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob


def shuffle_map_data(images, masks):
    data = tf.data.Dataset.from_tensor_slices((images, masks))
    #shuffling data
    data = data.shuffle(len(images))
    #we apply transformation to our dataset
    data = data.map(map_fn)
    return data

def split_data(files, masks, ratio1, ratio2):
    num_images = len(masks)
    
    #The number of images in our validation and test set
    val_test_size = int(num_images*ratio1)
    
    #array of files that contains the validation and test images
    val_test_images = files[:val_test_size]
    #array of files for the training images
    train_images = files[val_test_size:]
    #array of files that contains the validation and test maks
    val_test_masks = masks[:val_test_size]
    #array of files for the masks images
    train_masks = masks[val_test_size:]

    #The number that will split validation and test
    split = int(len(val_test_masks)*ratio2)
    #perform same as above except on the smaller validation and test
    val_masks = val_test_masks[split:]
    val_images = val_test_images[split:]
    test_masks = val_test_masks[:split]
    test_images = val_test_images[:split]
    return train_images, train_masks, val_masks, val_images, test_masks, test_images

#Converting image and mask files to data ararys
def map_fn(image_fp, mask_fp):
    #reading data from file and decoding
    image = tf.io.read_file(image_fp)
    image = tf.image.decode_jpeg(image, channels=3)
    #rezising all the images to (512, 512)
    image = tf.image.resize(image, (512, 512))
    image = tf.cast(image, tf.float32) /255.0
    
    #reading data from file and decoding
    mask = tf.io.read_file(mask_fp)
    mask = tf.image.decode_png(mask, channels=1)
    #rezising all the masks to (512, 512)
    mask = tf.image.resize(mask, (512, 512))
    #one hot encoding
    mask = mask == [0, 255]
    mask = tf.cast(mask, tf.uint8)
    return image, mask

#metrics used for model, smooth is used so we don't have a value of 0 for overlap
def dice_coef(true, pred, smooth=1):
    #true mask
    true1 = tf.keras.backend.flatten(true)
    #prediction mask
    pred1 = tf.keras.backend.flatten(pred)
    #Pixels that overlap and are equal in both images
    overlap = tf.keras.backend.sum(true1 * pred1)+smooth
    #Total number of pixels in the image
    totalPixels = (tf.keras.backend.sum(true1) + tf.keras.backend.sum(pred1))+smooth
    return (2 * overlap) / totalPixels

def convolution(inputs, filters):
    c1 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
    return tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(c1)

def unet():
    inputs = tf.keras.layers.Input(shape=(512, 512, 3))
    
    
    #Contraction path
    c1 = convolution(inputs, 4)
    
    
    c2 = tf.keras.layers.MaxPooling2D()(c1)
    c2 = convolution(c2, 8)
    
    c3 = tf.keras.layers.MaxPooling2D()(c2)
    c3 = convolution(c3, 16)
    
    c4 = tf.keras.layers.MaxPooling2D()(c3)
    c4 = convolution(c4, 32)
    
    c5 = tf.keras.layers.MaxPooling2D()(c4)
    c5 = convolution(c5, 64)
    
    #Expanding path
    c6 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides =(2, 2), padding='same')(c5)
    c6 = tf.keras.layers.concatenate([c6, c4])
    c6 = convolution(c6, 32)
    
    c7 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides =(2, 2),padding='same')(c6)
    c7 = tf.keras.layers.concatenate([c7, c3])
    c7 = convolution(c7, 16)

    c8 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides =(2, 2),padding='same')(c7)
    c8 = tf.keras.layers.concatenate([c8, c2])
    c8 = convolution(c8, 8)

    c9 = tf.keras.layers.Conv2DTranspose(4, (2, 2), strides =(2, 2),padding='same')(c8)
    c9 = tf.keras.layers.concatenate([c9, c1])
    c9 = convolution(c9, 4)
    
    #we use sigmoid because only black and white pixels
    outputs = tf.keras.layers.Conv2D(2, (1,1), activation='sigmoid')(c9)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def predictions(data, model, num=4):
    image_batch, mask_batch = next(iter(data.batch(num)))

    #input images
    plt.figure(figsize =(11, 11))
    for i in range(num):
        plt.subplot(2, num, i+1)
        #plot real images
        plt.imshow(image_batch[i])
        plt.axis('off')
    #prediction using our model
    predict = model.predict(image_batch)
    plt.figure(figsize = (11, 11))
    for i in range(num):
        plt.subplot(2, num, i+1)
        #plotting true mask
        plt.imshow(tf.argmax(mask_batch[i], axis=-1), cmap = 'gray')
        plt.axis('off')
    plt.figure(figsize = (11, 11))
    for i in range(num):
        plt.subplot(2, num, i+1)
        #plotting prediction mask
        plt.imshow(tf.argmax(predict[i], axis=-1), cmap = 'gray')
        plt.axis('off')
    print("Dice coefficient for images left to right")
    for i in range(num):
        text = "Dice Coefficient image {num} :{dice}"
        print(text.format(num = i+1, dice = dice_coef(tf.argmax(mask_batch[i], axis=-1), tf.argmax(predict[i], axis=-1)).numpy()))
def average_dice(data, model):
    image_batch, mask_batch = next(iter(data.batch(259)))
    #Prediction on all the images in the test set
    predict = model.predict(image_batch)
    sum = 0
    for i in range(259):
        sum = sum + dice_coef(tf.argmax(mask_batch[i], axis=-1), tf.argmax(predict[i], axis=-1)).numpy()
    #average of the dice coefficients over the test set
    print(sum/259)