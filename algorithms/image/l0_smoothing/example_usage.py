from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt

from image.l0_smoothing.l0_smoothing import l0_gradient_smoothing


def main():
    # Load an image and convert it to an array
    # Directly uses TensorFlow (but that still internally uses numpy and PIL)
    im = image.load_img(r'resources/wonka.png')
    input_image = image.img_to_array(im) / 255

    # Here we get the smoothed result
    # Optionally, you can provide additional arguments but usually the default arguments are good
    # See the function documentation and README to see what is available and what they do
    output_image = l0_gradient_smoothing(input_image)

    # Plot the results side by side for comparison:
    # Plot the input
    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.axis('off')
    plt.imshow(input_image)

    # Plot the smoothed output
    plt.subplot(1, 2, 2)
    plt.title('Smoothed Result')
    plt.axis('off')
    plt.imshow(output_image)

    plt.show()


if __name__ == '__main__':
    main()