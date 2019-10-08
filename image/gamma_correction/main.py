from gamma_correction import *
import cv2

# Driver Script
if __name__ == "__main__":
    # Load image using opencv.
    img = cv2.imread('input.png')

    # Call function to do gamma correction with gamma coef as 2.2.
    result = gamma_correction(img, 1.0/2.2)

    # Display the result.
    cv2.imshow('output', result)
    cv2.waitKey(0)
