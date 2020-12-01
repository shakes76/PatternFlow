# Simple test code to implement the adjust_sigmoid correction
# Use openCV to show the adjusted image
# ==========================================================
import cv2
from google.colab.patches import cv2_imshow

if __name__ == "__main__":
    # Load the test image.
    img = cv2.imread('Wally.jpg')

    # Call the sigmoid function.
    result = sigmoid(img)

    # Display the result.
    cv2_imshow(result)
