import matplotlib.pyplot as plt
from skimage import data
from adjust_sigmoid import adjust_sigmoid

if __name__ == "__main__":

    # Get the moon image from skimage
    image = data.moon()

    # Adjust the image with the sigmoid correction
    adjusted = adjust_sigmoid(image)

    # Plot the original and adjusted for comparision
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title("Original")
    fig.add_subplot(1, 2, 2)
    plt.imshow(adjusted, cmap=plt.cm.gray)
    plt.title("Adjusted")
    plt.show()
