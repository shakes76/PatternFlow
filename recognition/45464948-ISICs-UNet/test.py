import matplotlib.pyplot as plt
import tensorflow as tf

def plotResult(displaylist):
    plt.figure(20,20)
    title= ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(displaylist)):
        plt.subplot(1, len(displaylist), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(displaylist[i]))
        plt.axis('off')
    plt.show()