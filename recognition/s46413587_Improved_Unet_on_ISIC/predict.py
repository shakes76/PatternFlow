import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from random import sample, choice


import train
import dataset

#Colour Map to create the predicted image from the 2D pixel vector
def form_colourmap(prediction,mapping):
    h,w = prediction.shape
    colour_label = np.zeros((h,w,3),dtype=np.uint8)    
    colour_label = mapping[prediction]
    colour_label = colour_label.astype(np.uint8)
    return colour_label
    
#Use the model to predict the 2D pixel vector
def make_prediction(model,img_path,shape):
    img= img_to_array(load_img(img_path , target_size= shape))/255.
    img = np.expand_dims(img,axis=0)
    labels = model.predict(img)
    labels = np.argmax(labels[0],axis=2)
    return labels

#Produce 5 sample images with matching predictions and the true mask (from validation set)
for i in range(5):
    img_mask = choice(dataset.val_pair)
    img= img_to_array(load_img(img_mask[0] , target_size= (dataset.img_w,dataset.img_h)))
    gt_img = img_to_array(load_img(img_mask[1] , target_size= (dataset.img_w,dataset.img_h)))

    pred_label = make_prediction(train.model, img_mask[0], (dataset.img_w,dataset.img_h,3))

    pred_coloured = form_colourmap(pred_label,np.array(dataset.class_map))

    plt.figure(figsize=(15,15))
    plt.subplot(131);plt.title('Original Image')
    plt.imshow(img/255.)
    plt.subplot(132);plt.title('True labels')
    plt.imshow(gt_img/255.)
    plt.subplot(133)
    plt.imshow(pred_coloured/255., cmap='gray');plt.title('predicted labels')