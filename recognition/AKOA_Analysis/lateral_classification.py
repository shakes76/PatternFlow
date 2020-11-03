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
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy', 'mse'])

    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)

    #Plot accuracy and validation accuracy
    
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1.1])
    plt.legend(loc='lower right')    
    plt.savefig('graphs/accuracy.png')

    test_loss, test_accm, test_mse = model.evaluate(X_test,  y_test, verbose=2)
    
    print('test_loss:', test_loss)
    print('test_accm:', test_accm)
    
if __name__ == '__main__':
    main()