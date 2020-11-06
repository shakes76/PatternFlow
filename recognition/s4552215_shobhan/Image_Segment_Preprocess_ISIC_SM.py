# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:51:14 2020

@author: s4552215
"""


import Data_Load_ISIC_SM as IDL
from keras.utils import to_categorical


# To Process the lesion images and to split them into train,validation and test datasets
def proc_img(img_path,img_width,img_height):
    X_all = IDL.loading_img(img_path,img_width,img_height)
    train_split = int(round(0.6*len(X_all),0))
    test_val_split = int(round(0.2*len(X_all),0))
    XISIC_train = X_all[:train_split]
    XISIC_val = X_all[train_split:train_split + test_val_split]
    XISIC_test = X_all[train_split + test_val_split:train_split + 2*test_val_split]
    return (XISIC_train, XISIC_val, XISIC_test)

# To Process the segmentation images and to split them into train,validation and test datasets
def proc_seg(seg_path,img_width,img_height):
    Y_all = IDL.loading_seg(seg_path,img_width,img_height)
    train_split = int(round(0.6*len(Y_all),0))
    test_val_split = int(round(0.2*len(Y_all),0))
    YISIC_train = Y_all[:train_split]
    YISIC_val = Y_all[train_split:train_split + test_val_split]
    YISIC_test = Y_all[train_split + test_val_split:train_split + 2*test_val_split]
    return (YISIC_train, YISIC_val, YISIC_test)

# To Transform the segmentation images into categorical data
def pix_cat(y_train,y_val,y_test):
    y_train_sc = y_train//255
    y_val_sc = y_val//255
    y_test_sc = y_test//255
    y_train_cat = to_categorical(y_train_sc)
    y_val_cat = to_categorical(y_val_sc)
    y_test_cat = to_categorical(y_test_sc)
    return (y_train_cat, y_val_cat, y_test_cat)
