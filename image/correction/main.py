import matplotlib.pyplot as plt
from skimage import data
from PatternFlow.image.correction.correction import adjust_log


def main():
    img = data.moon()
    img_log = adjust_log(img)
    img_inv_log = adjust_log(img, inv=True)

    # config figure size
    fig = plt.figure(figsize=(10, 5))

    fig.add_subplot(1, 3, 1)
    plt.title("origin")
    plt.imshow(img, cmap=plt.cm.gray)

    fig.add_subplot(1, 3, 2)
    plt.title("log correction")
    plt.imshow(img_log, cmap=plt.cm.gray)

    fig.add_subplot(1, 3, 3)
    plt.title("inverse log correction")
    plt.imshow(img_inv_log, cmap=plt.cm.gray)
    # plt.show()
    plt.savefig("correction_result.png")


if __name__ == '__main__':
    main()
