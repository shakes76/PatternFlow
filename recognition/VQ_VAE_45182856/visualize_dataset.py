import sys
import cv2
import glob
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img_dir = sys.argv[1]
    path = glob.glob(img_dir + "\*.png")
    n_images = 9
    images = []
    i = 0
    for img in path:
        if i == n_images:
            break
        img_ = cv2.imread(img)
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY) # convert to greyscale
        images.append(img_)
        i += 1
    fig, axs = plt.subplots(nrows=int(n_images ** (1/2)), ncols=int(n_images ** (1/2)), figsize=(7, 7))
    for i in range(int(n_images ** (1/2))):
        for j in range(int(n_images ** (1/2))):
            axs[i, j].imshow(images[(i* int(n_images ** (1/2))) + j])
            axs[i, j].axis('off')
    plt.show()
    