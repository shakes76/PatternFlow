__author__ = "Vikram Rayakottai Niranjanvel (45484450)"
__date__ = "05/11/2019"

"""
Driver script to test the working of tensorflow implementation of the rescale_intensity module.
Licensing: Modeule was devloped for the Pattern recognition and image processing library. 
You are free to use the rescale_intensity modeule in any implementations. 

The script runs a couple of tests to verify the implementation of the rescale intensity modeule. 
The np library is used to compare the results of the implemented module.
"""

from rescale_intensity import rescale_intensity
import numpy as np
from  skimage.exposure import rescale_intensity as rescale_intensity_numpy
import matplotlib.pyplot as plt
from skimage import data


def test_function(array, in_range=None, out_range=None):
    """
    :param array (np array): Input array to be tested
    :param in_range:  "image"-> If the range is to be chosen from the image
                    "dtype"-> If the range of the dtype needs to be used
                    tuple-> User defined range
    :param out_range: "image"-> If the range is to be chosen from the image
                    "dtype"-> If the range of the dtype needs to be used
                    tuple-> User defined range
    return: (Bool)
    
    Checks if the results from np rescale intensity and rescale intensity tensorflow implementation match
    """
    
    
    #Different conditional statements to test based on the input parameters
    if in_range == None and out_range == None:
        tf_output = rescale_intensity(array)
        np_output = rescale_intensity_numpy(array)
        result = np.array_equal(np_output, tf_output)

    if in_range != None and out_range == None:
        tf_output = rescale_intensity(array, in_range=in_range)
        np_output = rescale_intensity_numpy(array, in_range=in_range)
        result = np.array_equal(np_output, tf_output)

    if in_range == None and out_range != None:
        tf_output = rescale_intensity(array, out_range=out_range)
        np_output = rescale_intensity_numpy(array, out_range=out_range)
        result = np.array_equal(np_output, tf_output)

    if in_range != None and out_range != None:
        tf_output = rescale_intensity(array, in_range=in_range, out_range=out_range)
        np_output = rescale_intensity_numpy(array, in_range=in_range, out_range=out_range)
        result = np.array_equal(np_output, tf_output)

    if result == True:
        return True, [tf_output, np_output]
    else:
        return False, [tf_output, np_output]

def main():
    #predefined testing conditions
    test_conditions = [
        {"input": np.array([1, 2, 3, 3], dtype="float"), "in_range": None, "out_range": None},
        {"input": np.array([1, 2, 3, 4], dtype="float"), "in_range": (0, 1), "out_range": None},
        {"input": np.array([51, 102, 153], dtype=np.uint8), "in_range": None, "out_range": None},
        {"input": np.array([51, 102, 153], dtype=np.uint8) * 1.0, "in_range": None, "out_range": None},
        {"input": np.array([51, 102, 153], dtype=np.uint8), "in_range": (0, 102), "out_range": None},
        {"input": np.array([51, 102, 153], dtype=np.uint8), "in_range": None, "out_range": (0, 102)}
    ]

    print("Test checks algorithm implementation with skimage api for similar results. It runs both algorithms and checks "
          +"if simlilar results are produced")
    
    #Runs through the test conditions and prints the failed outputs
    for test in test_conditions:
        result, output = test_function(test["input"], test["in_range"], test["out_range"])
        if result == True:
            print("input=", test["input"], "in range:", test["in_range"], "out range:", test["out_range"],
                  "\nResult: Passed")
            print(output[0])
        else:
            print("input=", test["input"], "in range:", test["in_range"], "out range:", test["out_range"],
                  "\nResult: Failed")
            print("Implementation", output[0])
            print("numpy implementation", output[1])

    #running a test on image data and plotting the results
    camera = data.camera()
    camera = np.array(camera)
    plt.imshow(camera)
    plt.imshow(rescale_intensity(camera, out_range=(0, 5)))


if __name__ == '__main__':
    main()
