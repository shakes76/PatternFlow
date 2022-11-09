from skimage import io
import cv2
import os
import numpy as np
import random
import tensorflow as tf


def load_data(file):
    '''
    read all pcs' file path
    pra: file 
    retrun file path <list>    
    '''
    path_list=[]
    for item in sorted(os.listdir(file)):
        try:
         #Remove (path) to remove hidden files that do not need to be processed
            if (item == ".DS_Store"):
                print (" find file .Ds_Store")
                os.remove(file + item)
            else:
                impath = file + item
                path_list.append(impath)
        except:
            pass
    return path_list
def read_data(data,colour):
    '''
    using cv2 to read picture as <256,256> vector
    
    pra: data <file_list>
         coulour, colour =1 is grey(1 channel) and colour =2 is coulour (3 channels)
    
    ruturn the img_list 
    
    '''
    img_list=[]
    if colour ==1:
        for i in data:
            img = cv2.imread(i,0)
            img=cv2.resize(img,(256,256))
            img_list.append(img)
            
    if colour ==2:
        for i in data:
            img = cv2.imread(i)
            img=cv2.resize(img,(256,256))
            img_list.append(img)
        
    return np.array(img_list)



def shuffle_data(X,y):
    '''
    
    shuffle the dataset then split the dataset to train,validation,test
    70% training data, 20% validation data, 20% testing data
    para:X and y
    return the X_train,X_val,X_test,y_train,y_val,y_test
    
    '''    
    combine = list(zip(X, y))
    random.shuffle(combine)
    X, y = zip(*combine)

    X_train = X[0:1816]
    X_val = X[1816:2205]
    X_test = X[2205:]
    
    
    y_train = y[0:1816]
    y_val = y[1816:2205]
    y_test = y[2205:]
    return X_train,X_val,X_test,y_train,y_val,y_test


def creat_mask(y):

    ###crete mask and in this mask there are two labels.

   
    y_train=[]
    for i in y:
        i = i /255.0
        i = np.where(i < 0.5, 0, i)
        i = np.where(i > 0.5, 1, i)
        label=tf.keras.utils.to_categorical(i, num_classes=2, dtype='float32')
        y_train.append(label)
    return y_train
