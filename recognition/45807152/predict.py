from utility import dice_coefficient, IoU
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def make_prediction(model, test_x, test_y, slice=(0, 1)):
    """
    Make predictions on test data using trained model.
    
    Specify slice of test data to predict.
    """
    # Get predictions on test data from trained model
    predictions = model.predict(test_x)

    first_test = slice[0]
    last_test = slice[1]
    figure_counter = 0

    for i in range(first_test, last_test):
        plt.figure(figure_counter)
        figure_counter += 1

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # Original Image Plot
        ax1.set_title("Original Image")
        ax1.axis("off")
        ax1.imshow(test_x[i])
        # Ground Truth Mask Plot
        ax2.set_title("Ground Truth Mask")
        ax2.axis("off")
        ax2.imshow(tf.argmax(test_y[i], axis=2), cmap='gray')
        # Get metrics
        pred_dice = np.around(dice_coefficient(
            test_y[i], predictions[i]).numpy(), 3)
        pred_iou = np.around(IoU(test_y[i], predictions[i]).numpy(), 3)
        
        ax3.set_title(f"Prediction")
        ax3.axis("off")
        ax3.text(0, 150, f"DSC={str(pred_dice)}, IoU={str(pred_iou)}")
        ax3.imshow(tf.argmax(predictions[i], axis=2), cmap='gray')

        plt.savefig(f"./figures/testComparison{i}.png")

    return
