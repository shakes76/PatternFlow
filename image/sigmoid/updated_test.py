# Test Sigmoid Correction according to the original adjust_sigmoid function result
# =======================
import numpy as np

def test_adjust_sigmoid_1x1_shape():
    """Check that the shape is maintained"""
    img = np.ones([1, 1])
    result = sigmoid(img, 1, 5)
    assert img.shape == result.shape


def test_adjust_sigmoid_cutoff_one():
    """Verifying the output with expected results for sigmoid correction
    with cutoff equal to one and gain of 5"""
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    expected = np.array([
        [  1,   1,   1,   2,   2,   2,   2,   2],
        [  3,   3,   3,   4,   4,   4,   5,   5],
        [  5,   6,   6,   7,   7,   8,   9,  10],
        [ 10,  11,  12,  13,  14,  15,  16,  18],
        [ 19,  20,  22,  24,  25,  27,  29,  32],
        [ 34,  36,  39,  41,  44,  47,  50,  54],
        [ 57,  61,  64,  68,  72,  76,  80,  85],
        [ 89,  94,  99, 104, 108, 113, 118, 123]], dtype=np.uint8)

    result = sigmoid(image, 1, 5)
    assert_array_equal(result, expected)


def test_adjust_sigmoid_cutoff_zero():
    """Verifying the output with expected results for sigmoid correction
    with cutoff equal to zero and gain of 10"""
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    expected = np.array([
        [127, 137, 147, 156, 166, 175, 183, 191],
        [198, 205, 211, 216, 221, 225, 229, 232],
        [235, 238, 240, 242, 244, 245, 247, 248],
        [249, 250, 250, 251, 251, 252, 252, 253],
        [253, 253, 253, 253, 254, 254, 254, 254],
        [254, 254, 254, 254, 254, 254, 254, 254],
        [254, 254, 254, 254, 254, 254, 254, 254],
        [254, 254, 254, 254, 254, 254, 254, 254]], dtype=np.uint8)

    result = sigmoid(image, 0, 10)
    assert_array_equal(result, expected)


def test_adjust_sigmoid_cutoff_half():
    """Verifying the output with expected results for sigmoid correction
    with cutoff equal to half and gain of 10"""
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    expected = np.array([
        [  1,   1,   2,   2,   3,   3,   4,   5],
        [  5,   6,   7,   9,  10,  12,  14,  16],
        [ 19,  22,  25,  29,  34,  39,  44,  50],
        [ 57,  64,  72,  80,  89,  99, 108, 118],
        [128, 138, 148, 158, 167, 176, 184, 192],
        [199, 205, 211, 217, 221, 226, 229, 233],
        [236, 238, 240, 242, 244, 246, 247, 248],
        [249, 250, 250, 251, 251, 252, 252, 253]], dtype=np.uint8)

    result = sigmoid(image, 0.5, 10)
    assert_array_equal(result, expected)


def test_adjust_inv_sigmoid_cutoff_half():
    """Verifying the output with expected results for inverse sigmoid
    correction with cutoff equal to half and gain of 10"""
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    expected = np.array([
        [253, 253, 252, 252, 251, 251, 250, 249],
        [249, 248, 247, 245, 244, 242, 240, 238],
        [235, 232, 229, 225, 220, 215, 210, 204],
        [197, 190, 182, 174, 165, 155, 146, 136],
        [126, 116, 106,  96,  87,  78,  70,  62],
        [ 55,  49,  43,  37,  33,  28,  25,  21],
        [ 18,  16,  14,  12,  10,   8,   7,   6],
        [  5,   4,   4,   3,   3,   2,   2,   1]], dtype=np.uint8)

    result = sigmoid(image, 0.5, 10, True)
    assert_array_equal(result, expected)
