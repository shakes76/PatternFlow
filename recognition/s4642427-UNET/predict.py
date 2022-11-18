import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def predict(model, results, X_test, Y_test):
    
    eval = model.evaluate(X_test, Y_test, batch_size=128)
    mask_prediction = model.predict(X_test)

    print("Dice Coefficient", eval[2])

    # LOSS GRAPH
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title("Loss (Blue) & Val_loss (Orange)")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # DICE COEFFICIENT GRAPH
    plt.plot(results.history['diceCoefficient'])
    plt.plot(results.history['val_diceCoefficient'])
    plt.title("Dice Coefficient (blue) & Val Dice Coefficient (orange)")
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.show()

    # Plot input image, mask and predicted mask
    for i in range(20,25):
        
        plt.figure(figsize=(10,10))
        
        # Plot the test image
        plt.subplot(1, 3, 1)
        plt.imshow(X_test[i])
        plt.title("Input Image")
        plt.axis("off")
        
        # Plot the test mask
        plt.subplot(1, 3, 2)
        plt.imshow(np.squeeze(Y_test[i]), cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis("off")
        
        # Plot the resultant mask
        plt.subplot(1, 3, 3)
        
        # Display 0 or 1 for classes
        prediction = tf.where(np.squeeze(mask_prediction[i]) > 0.5, 1.0, 0.0)
        plt.imshow(prediction, cmap='gray')
        plt.title("Resultant Mask")
        plt.axis("off")
        
        plt.show()