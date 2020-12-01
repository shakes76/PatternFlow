import sys
import time
import radon
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread
import skimage.transform as sk

def main(arglist):
    if (len(arglist) == 1):
        file_name = arglist[0]
        start_angle = 0
        end_angle = 180
        circle = True
    elif (len(arglist) == 3):
        file_name = arglist[0]
        start_angle = int(arglist[1])
        end_angle = int(arglist[2])
        circle = True
    elif (len(arglist) == 4):
        file_name = arglist[0]
        start_angle = int(arglist[1])
        end_angle = int(arglist[2])
        circle = bool(int(arglist[3]))
    else:
        print("Usage: test.py file_name {start_angle end_angle} {circle}")
        return
    
    # tensorflow eager execution setting
    #tf.compat.v1.disable_eager_execution()
    #sess = tf.compat.v1.Session()
    tf.executing_eagerly()
    
    # read image and convert to tensor
    image = imread(file_name, as_gray = True)
    tensorimage = tf.constant(image)
    
    # calculate radon transforms
    start = time.time()
    transformed = radon.radon(tensorimage, list(range(start_angle, end_angle)), circle)
    #transformed = transformed.eval(session=sess)
    end = time.time()
    port_runtime = end - start
    
    start = time.time()
    check = sk.radon(image, list(range(start_angle, end_angle)), circle)
    end = time.time()
    orig_runtime = end - start
    
    # print runtimes
    print("Port Runtime: " + str(port_runtime))
    print("Orig Runtime: " + str(orig_runtime))
    
    # plot images
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(transformed)
    plt.title("Port Radon Transform")
    plt.subplot(1, 3, 3)
    plt.imshow(check)
    plt.title("Original Radon Transform")
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])