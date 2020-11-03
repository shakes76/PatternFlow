import numpy as np
from sklearn.model_selection import train_test_split
import glob
import PIL
from PIL import Image
import matplotlib as ml
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

def train(images, shape, epochs):
    #Import the data

    filelist = glob.glob(images)
    image = np.array([np.array(Image.open(fname)) for fname in filelist])
    
    #Create label

    label = []

    #Populate label list

    for fname in filelist:
        if 'right' in fname.lower():
            label.append(0)
        elif 'R_I_G_H_T' in fname:
            label.append(0)
        elif 'left' in fname.lower():
            label.append(1)
        elif 'L_E_F_T' in fname:
            label.append(1)
    
if __name__ == '__main__':
    main()