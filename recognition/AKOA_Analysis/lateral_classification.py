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
            
    #Convert label list to numpy array

    label = np.array(label)

    #Split label and images into test and training sets

    X_train, X_test, y_train, y_test = train_test_split(image, label, test_size=0.25, random_state=42)
    
    #Create model

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2))

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
    
if __name__ == '__main__':
    main()