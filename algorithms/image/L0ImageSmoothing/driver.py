import argparse
import imageio
import matplotlib.pyplot as plt
import time

from l0_image_smoothing import l0_image_smoother


def main():
    """
    Main example driver for L0 Image Smoothing algorithm.

    General usage:
    img = imageio.imread('image.jpg')  # Or any other method to get numpy array.
    smoothed_image = l0_image_smoother(img, _lambda, kappa, beta_max)
    plt.imshow(smoothed_image)
    plt.axis('off')
    plt.savefig("smoothed.png", bbox_inches='tight')
    """
    parser = argparse.ArgumentParser(
        description="Tensorflow 2.0 L0 Gradient Norm image smoothing, v0.1. "
                    "Please see README.md for reference to original paper and matlab implementation.")
    parser.add_argument('-img',
                        help="Image file name or full path. "
                             "E.g., path/to/image.jpg, or image.jpg if same as working dir.")
    parser.add_argument('-l',
                        help="Smoothing weight, lambda. Ideally in range [1e-3, 1e-1]. Default=2e-2.",
                        default=2e-2)
    parser.add_argument('-k',
                        help="Parameter for update speed and update weight. Ideally (1, 2]. Default=2.0",
                        default=2.0)
    parser.add_argument('-b',
                        help="Beta max, upper bound for iteration calculation, beta * kappa. Default=1e5",
                        default=1e5)

    args = parser.parse_args()

    # Check that an image is provided
    if args.img is None:
        print("Error: Please provide image name, "
              "e.g., image.jpg if image in same working directly. "
              "Or provide full path.")
        return

    img = imageio.imread(args.img)
    _lambda = float(args.l)
    kappa = float(args.k)
    beta_max = float(args.b)

    print("L0 Smoothing algorithm started!")
    start = time.time()
    smoothed_image = l0_image_smoother(img, _lambda, kappa, beta_max)
    end = time.time()
    print(f"Image smoothed in {round((end-start)/60, 3)} minutes ({round(end-start, 3)} seconds)")
    plt.imshow(smoothed_image)
    plt.axis('off')
    plt.savefig("smoothed.png", bbox_inches='tight')
    print(f"Image saved in current dir as smoothed.png.")
    plt.show()


if __name__ == '__main__':
    main()

