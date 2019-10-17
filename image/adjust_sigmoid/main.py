import matplotlib.pyplot as plt
from skimage import data
from sigmoid import adjust_sigmoid

if __name__ == "__main__":

    image = data.moon()
    adjusted = adjust_sigmoid(image)

    fig = plt.figure()
    
    fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title("Original")
    
    fig.add_subplot(1, 2, 2)
    plt.imshow(adjusted, cmap=plt.cm.gray)
    plt.title("Adjusted")

    plt.show()
