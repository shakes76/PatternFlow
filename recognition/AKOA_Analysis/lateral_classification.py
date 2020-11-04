import matplotlib as ml
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

def train(shape, epochs, image_train, label_train, image_test, label_test):
    
    #Create model

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy', 'mse'])

    history = model.fit(image_train, label_train, epochs=epochs, validation_split=0.2)

    #Plot accuracy and validation accuracy
    
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1.1])
    plt.legend(loc='lower right')    
    plt.savefig('graphs/accuracy.png')

    test_loss, test_accm, test_mse = model.evaluate(image_test,  label_test, verbose=2)
    
    print('test_loss:', test_loss)
    print('test_accm:', test_accm)
    
if __name__ == '__main__':
    main()