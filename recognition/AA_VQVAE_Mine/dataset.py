import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical ,Sequence
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import tensorflow as tf

import os
from PIL import Image
from glob import glob
from pathlib import Path
from random import sample, choice
import shutil

img_h = 432
img_w = 288
b_size = 32


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

(len(train_imgs),len(train_labels)), (len(test_imgs),len(test_labels)) , (len(val_imgs),len(val_labels))

def make_pair(img,label,dataset):
    pairs = []
    for im in img:
        pairs.append((im , dataset / label / (im.stem +"_segmentation.png")))
    
    return pairs

train_pair = make_pair(train_imgs, "ISIC-2017_Training_Truth", im_root)
test_pair = make_pair(test_imgs, "ISIC-2017_Test_Truth", im_root)
val_pair = make_pair(val_imgs, "ISIC-2017_Validation_Truth", im_root)

temp = choice(train_pair)
img = img_to_array(load_img(temp[0], target_size=(img_w,img_h)))
mask = img_to_array(load_img(temp[1], target_size = (img_w,img_h)))
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(img/255)
plt.subplot(122)
plt.imshow(mask/255)

#plt.show()

class_map = [(255),(0)]

def assert_map_range(mask,class_map):
    mask = mask.astype("uint8")
    for j in range(img_w):
        for k in range(img_h):
            assert mask[j][k] in class_map , tuple(mask[j][k])

def form_2D_label(mask,class_map):
    mask = mask.astype("uint8")
    label = np.zeros(mask.shape[:2],dtype= np.uint8)
    
    for i, rgb in enumerate(class_map):
        label[(mask == rgb).all(axis=2)] = i
    
    return label

lab = form_2D_label(mask,class_map)
np.unique(lab,return_counts=True)

class DataGenerator(Sequence):
    'Generates data for Keras'
    
    def __init__(self, pair, class_map, batch_size=16, dim=(224,224,3), shuffle=True):
        'Initialization'
        self.dim = dim
        self.pair = pair
        self.class_map = class_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.pair) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.pair))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_imgs = list()
        batch_labels = list()

        # Generate data
        for i in list_IDs_temp:
            # Store sample
            img = load_img(self.pair[i][0] ,target_size=self.dim)
            img = img_to_array(img)/255.
            batch_imgs.append(img)

            label = load_img(self.pair[i][1],target_size=self.dim)
            label = img_to_array(label)
            label = form_2D_label(label,self.class_map)
            label = to_categorical(label , num_classes = 2)
            batch_labels.append(label)
            
        return np.array(batch_imgs) ,np.array(batch_labels)

train_generator = DataGenerator(train_pair+test_pair,class_map,b_size, dim=(img_w,img_h,3) ,shuffle=True)
train_steps = train_generator.__len__()

X,Y = train_generator.__getitem__(1)
print(Y.shape)

val_generator = DataGenerator(val_pair, class_map, batch_size=4, dim=(img_w,img_h,3) ,shuffle=True)
val_steps = val_generator.__len__()


print("no errors")


'''

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