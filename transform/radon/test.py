import sys
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
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    #tf.executing_eagerly()
    
    # read image and convert to tensor
    image = imread(file_name, as_gray = True)
    tensorimage = tf.constant(image)
    
    # calculate radon transforms
    transformed = radon.radon(tensorimage, list(range(start_angle, end_angle)), circle)
    check = sk.radon(image, list(range(start_angle, end_angle)), circle)
    
    # plot images
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(transformed.eval(session=sess))
    plt.subplot(1, 3, 3)
    plt.imshow(check)
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])