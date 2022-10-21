import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
def dice_coefficient(a, b):
    """
    Dice Coefficient function from : https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Used to determine how closely two set overlap each other. In this case we use it to see how close the
    predicted mask matches the ground truth mask.
    """
   
    
    a = K.flatten(a)
    b = K.flatten(b)
    a_union_b = K.sum(a * b)
    mag_a = K.sum(a)
    mag_b = K.sum(b)
    
    return (2.0 * a_union_b) / (mag_a + mag_b)

def dice_coefficient_loss(truth, predition):
    """
    Loss function as described in the Improved Unet paper.
    """
    return 1 - dice_coefficient(truth, predition)

def plot_accuracy(history):
    """
    Plots the accuracy of the model throughout the training process.
    """
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Test and Validation Accuracy")
    plt.legend(loc='lower right')
    plt.savefig("./images/accuracy.png")

def plot_loss(history):
    """
    Plots the loss of the model throughout a training session.
    """
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Test and Validation Loss")
    plt.legend(loc='lower right')
    plt.savefig("./images/loss.png")

