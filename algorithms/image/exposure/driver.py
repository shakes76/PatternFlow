import torch
import matplotlib.pyplot as plt
from exposure import (  # type: ignore
    histogram,
    equalize_hist,
    cumulative_distribution,
    adjust_gamma,
)
from skimage import data


def histogram_demo():
    image = torch.tensor(data.camera())
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax1.set_title("original image")
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    hist, bins = histogram(image)
    ax2.set_title("histogram of image")
    ax2.hist(hist, bins=bins)
    f.show()


def cumulative_distribution_demo():
    image = torch.tensor(data.coffee())
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax1.set_title("original image")
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    hist, bins = cumulative_distribution(image)
    ax2.plot(bins, hist, 'b-')
    ax2.set_title("cdf of image")
    f.show()


def equalize_hist_demo():
    image = torch.tensor(data.camera())
    equ_image = equalize_hist(image)
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("original image")
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.imshow(image)
    ax2.set_title("after equalization")
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.imshow(equ_image)
    f.show()


def adjust_gamma_demo():
    image = torch.tensor(data.astronaut())
    shift_left_img = adjust_gamma(image, 1.5)
    shift_right_img = adjust_gamma(image, 0.5)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_title("original image")
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.imshow(image)
    ax2.set_title("adjust gamma 1.5")
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.imshow(shift_left_img)
    ax3.set_title("adjust gamma 0.5")
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.imshow(shift_right_img)
    f.show()


def main():
    histogram_demo()
    cumulative_distribution_demo()
    equalize_hist_demo()
    adjust_gamma_demo()
    plt.show()


if __name__ == "__main__":
    main()
