'''
    File name: ImprovedUnet_OASIS.py
    Author: Bin Lyu(45740165)
    Date created: 10/23/2020
    Date last modified: 11/01/2020
    Python Version: 4.7.4
'''
import tensorflow as tf
from os import listdir
from numpy import asarray
from PIL import Image
from matplotlib import pyplot
import glob
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from IPython.display import clear_output

# data processing, save the images to the corresponding dataset
train_images = sorted(glob.glob("/Users/annalyu/Desktop/Data_Science/2020.2/COMP3710/Ass/Ass3Data/keras_png_slices_data/keras_png_slices_train/*.png"))
train_masks = sorted(glob.glob("/Users/annalyu/Desktop/Data_Science/2020.2/COMP3710/Ass/Ass3Data/keras_png_slices_data/keras_png_slices_seg_train/*.png"))
test_images = sorted(glob.glob("/Users/annalyu/Desktop/Data_Science/2020.2/COMP3710/Ass/Ass3Data/keras_png_slices_data/keras_png_slices_test/*.png"))
test_masks = sorted(glob.glob("/Users/annalyu/Desktop/Data_Science/2020.2/COMP3710/Ass/Ass3Data/keras_png_slices_data/keras_png_slices_seg_test/*.png"))
val_images = sorted(glob.glob("/Users/annalyu/Desktop/Data_Science/2020.2/COMP3710/Ass/Ass3Data/keras_png_slices_data/keras_png_slices_validate/*.png"))
val_masks = sorted(glob.glob("/Users/annalyu/Desktop/Data_Science/2020.2/COMP3710/Ass/Ass3Data/keras_png_slices_data/keras_png_slices_seg_validate/*.png"))

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_masks))

#shuffle the dataset
train_ds = train_ds.shuffle(len(train_images))
val_ds = val_ds.shuffle(len(val_images))
test_ds = test_ds.shuffle(len(test_images))

def decode_png(file_path):
    #decode and reshape the images to 256*256
    png = tf.io.read_file(file_path)
    png = tf.image.decode_png(png, channels=1)
    png = tf.image.resize(png, (256, 256))
    return png

def process_path(image_fp, mask_fp):
    image = decode_png(image_fp)
    image = tf.cast(image, tf.float32)/ 255.0

    mask = decode_png(mask_fp)
    mask = mask == [0, 85, 170, 255]
    mask = tf.cast(mask, tf.float32)
    return image, mask

train_ds = train_ds.map(process_path)
val_ds = val_ds.map(process_path)
test_ds = test_ds.map(process_path)

def display(display_list):
    #display images
    plt.figure(figsize=(10, 10))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')
    plt.show()

#show one image for example
for image, mask in train_ds.take(1):
    display([tf.squeeze(image), tf.argmax(mask, axis=-1)])

#develop the unet model according to the paper(just change the 3D structure to 2D)
#use leakReLu(0.01) according to the paper mentioned
def unet_model(output_channels, f=64):
    inputs = tf.keras.layers.Input(shape=(256,256,1))
    d1_conv = tf.keras.layers.Conv2D(f, 3, padding='same')(inputs)
    d1_conv = tf.keras.layers.LeakyReLU(0.01)(d1_conv)

    #context modules with two 3*3 convolutional layers and a dropout layers in between
    d1 = tf.keras.layers.Conv2D(f, 3, padding='same')(d1_conv)
    d1 = tf.keras.layers.LeakyReLU(0.01)(d1)
    d1 = tf.keras.layers.Dropout(0.3)(d1)
    d1 = tf.keras.layers.Conv2D(f, 3, padding='same')(d1)
    d1 = tf.keras.layers.LeakyReLU(0.01)(d1)
    #d1 = d1_conv + d1
    d1 = tf.keras.layers.add([d1_conv, d1])

    #stride two convolutional layer to reduce the resolution
    d2_conv = tf.keras.layers.Conv2D(2*f, 3, 2, padding='same')(d1)
    d2_conv = tf.keras.layers.LeakyReLU(0.01)(d2_conv)
    #context modules
    d2 = tf.keras.layers.Conv2D(2*f, 3, padding='same')(d2_conv)
    d2 = tf.keras.layers.LeakyReLU(0.01)(d2)
    d2 = tf.keras.layers.Dropout(0.3)(d2)
    d2 = tf.keras.layers.Conv2D(2*f, 3, padding='same')(d2)
    d2 = tf.keras.layers.LeakyReLU(0.01)(d2)
    #element-wise sum
    d2 = tf.keras.layers.add([d2_conv, d2])

    #stride two convolutional layer to reduce the resolution
    d3_conv = tf.keras.layers.Conv2D(4*f, 3, 2, padding='same')(d2)
    d3_conv = tf.keras.layers.LeakyReLU(0.01)(d3_conv)
    #context modules
    d3 = tf.keras.layers.Conv2D(4*f, 3, padding='same')(d3_conv)
    d3 = tf.keras.layers.LeakyReLU(0.01)(d3)
    d3 = tf.keras.layers.Dropout(0.3)(d3)
    d3 = tf.keras.layers.Conv2D(4*f, 3, padding='same')(d3)
    d3 = tf.keras.layers.LeakyReLU(0.01)(d3)
    #element-wise sum
    d3 = tf.keras.layers.add([d3_conv, d3])
    
    #stride two convolutional layer to reduce the resolution
    d4_conv = tf.keras.layers.Conv2D(8*f, 3, 2, padding='same')(d3)
    d4_conv = tf.keras.layers.LeakyReLU(0.01)(d4_conv)
    #context modules
    d4 = tf.keras.layers.Conv2D(8*f, 3, padding='same')(d4_conv)
    d4 = tf.keras.layers.LeakyReLU(0.01)(d4)
    d4 = tf.keras.layers.Dropout(0.3)(d4)
    d4 = tf.keras.layers.Conv2D(8*f, 3, padding='same')(d4)
    d4 = tf.keras.layers.LeakyReLU(0.01)(d4)
    #element-wise sum
    d4 = tf.keras.layers.add([d4_conv, d4])
    
    #stride two convolutional layer to reduce the resolution
    d5_conv = tf.keras.layers.Conv2D(16*f, 3, 2, padding='same')(d4)
    d5_conv = tf.keras.layers.LeakyReLU(0.01)(d5_conv)
    #context modules
    d5 = tf.keras.layers.Conv2D(16*f, 3, padding='same')(d5_conv)
    d5 = tf.keras.layers.LeakyReLU(0.01)(d5)
    d5 = tf.keras.layers.Dropout(0.3)(d5)
    d5 = tf.keras.layers.Conv2D(16*f, 3, padding='same')(d5)
    d5 = tf.keras.layers.LeakyReLU(0.01)(d5)
    #element-wise sum
    d5 = tf.keras.layers.add([d5_conv, d5])

    #upsampling module-u4
    u4 = tf.keras.layers.UpSampling2D()(d5)
    u4 = tf.keras.layers.Conv2D(8*f, 3, padding='same')(u4)
    u4 = tf.keras.layers.LeakyReLU(0.01)(u4)
    #concatenate-u4
    u4 = tf.keras.layers.concatenate([u4, d4])
    #localization-u4
    u4 = tf.keras.layers.Conv2D(8*f, 3, padding='same')(u4)
    u4 = tf.keras.layers.LeakyReLU(0.01)(u4)
    u4 = tf.keras.layers.Conv2D(8*f, 1, padding='same')(u4)
    u4 = tf.keras.layers.LeakyReLU(0.01)(u4)

    #upsampling-u3
    u3 = tf.keras.layers.UpSampling2D()(u4)
    u3 = tf.keras.layers.Conv2D(4*f, 3, padding='same')(u3)
    u3 = tf.keras.layers.LeakyReLU(0.01)(u3)
    #concatenate-u3
    u3 = tf.keras.layers.concatenate([u3, d3])
    #localization-u3
    u3 = tf.keras.layers.Conv2D(4*f, 3, padding='same')(u3)
    u3 = tf.keras.layers.LeakyReLU(0.01)(u3)
    u3 = tf.keras.layers.Conv2D(4*f, 1, padding='same')(u3)
    u3 = tf.keras.layers.LeakyReLU(0.01)(u3)
    #segmentation-u3
    print("output_channels",output_channels)
    seg3 = tf.keras.layers.Conv2D(output_channels, 1, padding='same')(u3)
    seg3 = tf.keras.layers.LeakyReLU(0.01)(seg3)
    seg3 = tf.keras.layers.UpSampling2D()(seg3)

    #upsampling-u2
    u2 = tf.keras.layers.UpSampling2D()(u3)
    u2 = tf.keras.layers.Conv2D(2*f, 3, padding='same')(u2)
    u2 = tf.keras.layers.LeakyReLU(0.01)(u2)
    #concatenate-u2
    u2 = tf.keras.layers.concatenate([u2, d2])
    #localization-u2
    u2 = tf.keras.layers.Conv2D(2*f, 3, padding='same')(u2)
    u2 = tf.keras.layers.LeakyReLU(0.01)(u2)
    u2 = tf.keras.layers.Conv2D(2*f, 1, padding='same')(u2)
    u2 = tf.keras.layers.LeakyReLU(0.01)(u2)
    #segmentation-u2
    seg2 = tf.keras.layers.Conv2D(output_channels, 1, padding='same')(u2)
    seg2 = tf.keras.layers.LeakyReLU(0.01)(seg2)
    seg_sum = tf.keras.layers.add([seg3, seg2])
    seg_sum = tf.keras.layers.UpSampling2D()(seg_sum)

    #upsampling-u1
    u1 = tf.keras.layers.UpSampling2D()(u2)
    u1 = tf.keras.layers.Conv2D(f, 3, padding='same')(u1)
    u1 = tf.keras.layers.LeakyReLU(0.01)(u1)
    #concatenate-u1
    u1 = tf.keras.layers.concatenate([u1, d1])
    #localization-u1
    u1 = tf.keras.layers.Conv2D(f, 3, padding='same')(u1)
    u1 = tf.keras.layers.LeakyReLU(0.01)(u1)
    u1 = tf.keras.layers.Conv2D(f, 1, padding='same')(u1)
    u1 = tf.keras.layers.LeakyReLU(0.01)(u1)
    #segmentation-u1
    seg1 = tf.keras.layers.Conv2D(output_channels, 1, padding='same')(u1)
    seg1 = tf.keras.layers.LeakyReLU(0.01)(seg1)
    seg_sum = tf.keras.layers.add([seg1, seg_sum])
    outputs = tf.keras.layers.Conv2D(output_channels, 1, activation='softmax')(seg_sum)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

#compile and fit the model
model = unet_model(4, f=4)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds.batch(10),epochs=10,validation_data=val_ds.batch(10))

#visulization the predict result
def show_predictions(ds, num=1):
    for image, mask in ds.take(num):
        pred_mask = model.predict(image[tf.newaxis, ...])
        pred_mask = tf.argmax(pred_mask[0], axis=-1)
        display([tf.squeeze(image), tf.argmax(mask, axis=-1), pred_mask])

#dice similarity coefficient
def dice_coefficient(y_true, y_pred, smooth = 1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

#dice similarity coefficient loss
def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

#prediction fuction, get predict and true value for dice calculating
def prediction(ds):
    pred = []
    true = []
    for image, mask in ds:
        pred_mask = model.predict(image[tf.newaxis, ...])
        pred.append(pred_mask)
        true_mask = tf.cast(mask, tf.float32)
        true.append(true_mask)
    return pred, true

pred, true = prediction(test_ds)
dice = dice_coefficient(true, pred)
print("Dice similarity coefficient is: ", dice) 

#call show_predictions function to check the final result
show_predictions(test_ds, 3)