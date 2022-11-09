"""
Laterality classification of the OAI AKOA knee data set. This is a possible solution to Task 2.
Run this code as the driver script.

@author Jonathan Godbold, s4533974.

Usage of this file is strictly for The University of Queensland.
Date: 27/10/2020.

Description:
Plots the results from the trained model.
"""

# Import relevant libraries.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plotResults(model, test_images, test_images_y):
    # Plot a graph accuracy vs validation accuracy.
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    # Summarize history for loss.
    plt.plot(history.history['loss']) 
    plt.plot(history.history['val_loss']) 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()

    # Demo a test group.
    demo_images_1 = test_images[0:10]
    demo_labels_1 = test_images_y[0:10]

    # Plot the images of the knee.
    plt.figure(figsize=(30,30))
    fig,ax=plt.subplots(3,3)
    lister = list(unique_list) # List[91] is the first person in the test group.
    fig.suptitle("Paitient ID:" + lister[91])

    index=1

    for m in range(3):
        for n in range(3):
                ax[m,n].imshow(demo_images_1[index])
                index += 1
    plt.show()
    print("From the files, all of these images are from the same person and they are all of the left knee.")
    scores = model.evaluate(demo_images_1, demo_labels_1, verbose = 1)
    pred = model.predict(demo_images_1)
    print("From above, we can see that the model accurately identified all 9 images.")

    # Multiple different images (left and right)
    print(test_images_y[1000:1010])
    # This time we have 2 patients, one with a left knee, one with a right.
    demo_images_2 = test_images[1000:1010]
    demo_labels_2 = test_images_y[1000:1010]

    plt.figure(figsize=(30,30))
    fig,ax=plt.subplots(3,3)

    index=1

    for m in range(3):
        for n in range(3):
            ax[m,n].imshow(demo_images_2[index])
            index += 1
    plt.show()
    print("From the files, we have 2 different people one of a left knee and the other for a right knee.")
    scores = model.evaluate(demo_images_2, demo_labels_2, verbose = 1)
    pred = model.predict(demo_images_2)
    print("From above, we can see that the model accurately identified all 9 images.")
