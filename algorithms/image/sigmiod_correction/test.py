import numpy as np
from sigmiod_correction.sigmoid import adjust_sigmoid
from numpy.testing import assert_array_equal
import cv2

if __name__ == '__main__':
    # Verifying the output with expected results for sigmoid correction
    # with cutoff equal to one and gain of 5
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    expected = np.array([
        [1, 1, 1, 2, 2, 2, 2, 2],
        [3, 3, 3, 4, 4, 4, 5, 5],
        [5, 6, 6, 7, 7, 8, 9, 10],
        [10, 11, 12, 13, 14, 15, 16, 18],
        [19, 20, 22, 24, 25, 27, 29, 32],
        [34, 36, 39, 41, 44, 47, 50, 54],
        [57, 61, 64, 68, 72, 76, 80, 85],
        [89, 94, 99, 104, 108, 113, 118, 123]], dtype=np.uint8)

    result = adjust_sigmoid(image, 1, 5)
    assert_array_equal(result, expected)
    print("The output is equal to the expected results")

    # try to test a real image
    img = cv2.imread('uni.jpg')
    # Call the sigmoid function.
    result = adjust_sigmoid(img)
    # Display the result.
    cv2.imwrite('sigmoid_uni_img.jpg', result)
    cv2.imshow("sigmoid_uni_img", result)
    cv2.waitKey()