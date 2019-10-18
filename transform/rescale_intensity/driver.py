from numpy_imp import rescale_tf
import numpy as np
from  skimage.exposure import rescale_intensity as rescale_intensity_numpy
import matplotlib.pyplot as plt
from skimage import data


def test_function(array, in_range=None, out_range=None):
    if in_range == None and out_range == None:
        tf_output = rescale_tf(array)
        np_output = rescale_intensity(array)
        result = np.array_equal(np_output, tf_output)

    if in_range != None and out_range == None:
        tf_output = rescale_tf(array, in_range=in_range)
        np_output = rescale_intensity(array, in_range=in_range)
        result = np.array_equal(np_output, tf_output)

    if in_range == None and out_range != None:
        tf_output = rescale_tf(array, out_range=out_range)
        np_output = rescale_intensity(array, out_range=out_range)
        result = np.array_equal(np_output, tf_output)

    if in_range != None and out_range != None:
        tf_output = rescale_tf(array, in_range=in_range, out_range=out_range)
        np_output = rescale_intensity(array, in_range=in_range, out_range=out_range)
        result = np.array_equal(np_output, tf_output)

    if result == True:
        return True, [tf_output, np_output]
    else:
        return False, [tf_output, np_output]

def main():
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

    camera = data.camera()
    camera = np.array(camera)
    plt.imshow(camera)
    plt.imshow(rescale_intensity(camera, out_range=(0, 5)))


if __name__ == '__main__':
    main()