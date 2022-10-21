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

def plot_metrics(history):
    """
    Plots the dice coefficient and the loss of the model throughout the training process.
    """
    plt.figure(1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend(loc='lower right')
    plt.savefig("./report-images/loss.png")

    plt.figure(2)
    plt.plot(history.history['dice_coefficient'], label='Training Accuracy')
    plt.plot(history.history['val_dice_coefficient'], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Similarity")
    plt.title("Training and Validation Accuracy")
    plt.legend(loc='lower right')
    plt.savefig("./report-images/dice_accuracy.png")

   

