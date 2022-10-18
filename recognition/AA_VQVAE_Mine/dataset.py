import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical ,Sequence
import tensorflow_probability as tfp
import tensorflow as tf

import os
from PIL import Image
from glob import glob
from pathlib import Path
from random import sample, choice

img_w = 4288
img_h = 2848
b_size = 32
partition = {}
labels = {}

im_root = Path(os.path.join(os.getcwd(), "recognition\AA_VQVAE_Mine\DataSets\ISIC"))

paths = [
    "ISIC-2017_Training_Data",
    "ISIC-2017_Training_Truth",
    "ISIC-2017_Test_Data",
    "ISIC-2017_Test_Truth",
    "ISIC-2017_Validation_Data",
    "ISIC-2017_Validation_Truth"
]
'''
for p in paths:
    folder_fp = os.path.join(im_root, p)
    for im_fn in os.listdir(folder_fp):
        im_fp = os.path.join(folder_fp, im_fn)
        new_im = os.path.join(folder_fp, im_fn)
        im = Image.open(new_im).convert("L")
        im.save(new_im)
'''
train_imgs = list((im_root / "ISIC-2017_Training_Data").glob("*.jpg"))
train_labels = list((im_root / "ISIC-2017_Training_Truth").glob("*.png"))
test_imgs = list((im_root / "ISIC-2017_Test_Data").glob("*.jpg"))
test_labels = list((im_root / "ISIC-2017_Test_Truth").glob("*.png"))
val_imgs = list((im_root / "ISIC-2017_Validation_Data").glob("*.jpg"))
val_labels = list((im_root / "ISIC-2017_Validation_Truth").glob("*.png"))

(len(train_imgs),len(train_labels)), (len(val_imgs),len(val_labels)) , (len(test_imgs),len(test_labels))

def make_pair(img,label,dataset):
    pairs = []
    for im in img:
        pairs.append((im , dataset / label / (im.stem +"_segmentation.png")))
    
    return pairs

train_pair = make_pair(train_imgs, "ISIC-2017_Training_Truth", im_root)
test_pair = make_pair(val_imgs, "ISIC-2017_Test_Truth", im_root)
val_pair = make_pair(test_imgs, "ISIC-2017_Validation_Truth", im_root)

temp = choice(train_pair)
img = img_to_array(load_img(temp[0], target_size=(img_w,img_h)))
mask = img_to_array(load_img(temp[1], target_size = (img_w,img_h)))
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(img/255)
plt.subplot(122)
plt.imshow(mask/255)

plt.show()

class_map = []
for index,item in class_map_df.iterrows():
    class_map.append(np.array([item['b'], item['w']]))
    

'''
seed = 909 # (IMPORTANT) to transform image and corresponding mask with same augmentation parameter.
image_datagen = ImageDataGenerator(width_shift_range=0.1,
                 height_shift_range=0.1,
                 preprocessing_function = image_preprocessing) # custom fuction for each image you can use resnet one too.
mask_datagen = ImageDataGenerator(width_shift_range=0.1,
                 height_shift_range=0.1,
                 preprocessing_function = mask_preprocessing)  # to make mask as feedable formate (256,256,1)

image_generator =image_datagen.flow_from_directory(os.path.join(im_root, "ISIC-2017_Training_Data"),
                                                    class_mode=None, seed=seed)

mask_generator = mask_datagen.flow_from_directory(os.path.join(im_root, "ISIC-2017_Training_Truth"),
                                                   class_mode=None, seed=seed)
'''

print("no errors")


'''
im_root = path = os.path.join(os.getcwd(), "recognition\AA_VQVAE_Mine\DataSets\AD_NC")


training_set = tf.keras.utils.image_dataset_from_directory(
                        os.path.join(im_root,"train"),
                        labels='inferred',
                        label_mode='int',
                        color_mode='grayscale',
                        image_size=(image_width, image_height),
                        batch_size = None,
                        shuffle=True,
                        seed=46,
                        validation_split=0.3,
                        subset='training',
                        interpolation='bilinear',
                        crop_to_aspect_ratio=True
                    )

validation_set = tf.keras.utils.image_dataset_from_directory(
                        os.path.join(im_root,"train"),
                        labels='inferred',
                        label_mode='int',
                        color_mode='grayscale',
                        image_size=(image_width, image_height),
                        batch_size = None,
                        shuffle=True,
                        seed=46,
                        validation_split=0.3,
                        subset='validation',
                        interpolation='bilinear',
                        crop_to_aspect_ratio=True
                    )

test_set = tf.keras.utils.image_dataset_from_directory(
                    os.path.join(im_root,"test"),
                    labels='inferred',
                    label_mode='int',
                    color_mode='grayscale',
                    image_size=(image_width, image_height),
                    batch_size = None,
                    shuffle=True,
                    seed=46,
                    interpolation='bilinear',
                    crop_to_aspect_ratio=True
                )

class_names = training_set.class_names
#print(class_names)


"""Convert images to floating point with the range [0.5, 0.5]"""
(x_train, y_train) = zip(*training_set)
#x_train = np.expand_dims(x_train, -1)
x_train = np.asarray(x_train)
x_train_scaled = (x_train / 255.0) - 0.5
(x_val,y_val) = zip(*validation_set)
#x_val = np.expand_dims(x_val, -1)
x_val = np.asarray(x_val)
x_val_scaled = (x_val / 255.0) - 0.5
(x_test,y_test) = zip(*test_set)
#x_test = np.expand_dims(x_test, -1)
x_test = np.asarray(x_test)
x_test_scaled = (x_test / 255.0) - 0.5


data_variance = np.var(x_train / 255.0)



#And plot images
plt.figure(figsize=(10, 10))
for images, labels in training_set.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"),cmap='gray')
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()

import pickle

# example, replace with your result
filename = "resulta.pickle"
with open(filename, "wb") as file:
    pickle.dump(x_train, file)

filename = "resultb.pickle"
with open(filename, "wb") as file:
    pickle.dump(data_variance, file)

filename = "resultc.pickle"
with open(filename, "wb") as file:
    pickle.dump(x_test_scaled, file)
    '''