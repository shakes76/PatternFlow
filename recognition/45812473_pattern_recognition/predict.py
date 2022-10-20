from matplotlib import pyplot as plt
import tensorflow as tf
from utils import dice_similarity
import numpy

"""
Shows the first 5 predictions using the model and the test set
"""
def predictions(model, xtest, ytest):

    # Predict the masks 
    predictions = model.predict(xtest)
    fig = plt.figure()
    count = 1

    for i, prediction in enumerate(predictions):
        if count <= 15:
            # Plot the original image
            ax = plt.subplot(5, 3, count)
            if count <= 3:
                ax.set_title("Original Image")
            plt.axis("off")
            plt.imshow(xtest[i], cmap='gray')
            count = count + 1

            # Plot the original mask
            ax = plt.subplot(5, 3, count)
            if count <= 3:
                ax.set_title("Original Mask")
            plt.axis("off")
            plt.imshow(tf.argmax(ytest[i], axis=2), cmap='gray')
            count = count + 1

            # Plot the predicted mask
            ax = plt.subplot(5, 3, count)
            if count <= 3:
                ax.set_title("Predicted Mask")
            plt.axis("off")
            plt.imshow(tf.argmax(prediction, axis=2), cmap='gray')
            count = count + 1

    plt.show()
    fig.savefig("Predictions")
    return

def test_dsc(model, xtest, ytest):
    predictions = model.predict(xtest)

    dsc = 0
    for i, prediction in enumerate(predictions):
        dsc += dice_similarity(ytest[i], prediction)
    dsc = dsc / predictions.shape[0]
    dsc = dsc.numpy()

    print("Dice similarity coefficient of the test set is", dsc)
    return