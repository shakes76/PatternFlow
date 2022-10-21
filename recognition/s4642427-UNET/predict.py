import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def predict(model, X_test, Y_test):
    
    mask_prediction = model.predict(X_test)

    # Plot the original image, ground truth and result from the network.
    for i in range(5):
        
        plt.figure(figsize=(10,10))
        
        # Plot the test image
        plt.subplot(1, 3, 1)
        plt.imshow(X_test[i])
        plt.title("Image")
        plt.axis("off")
        
        # Plot the test mask
        plt.subplot(1, 3, 2)
        plt.imshow(np.squeeze(Y_test[i]))
        plt.title("Mask")
        plt.axis("off")
        
        # Plot the resultant mask
        plt.subplot(1, 3, 3)
        
        # Display 0 or 1 for classes
        prediction = tf.where(np.squeeze(mask_prediction[i]) > 0.5, 1.0, 0.0)
        plt.imshow(prediction)
        plt.title("Predicted Mask")
        plt.axis("off")
        
        plt.show()