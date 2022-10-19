import tensorflow as tf
import os
from tensorflow import keras
from train import StyleGAN

# Input File Locations
INPUT_IMAGES_PATH = "keras_png_slices"      # path for input images (non-null)

# INPUT_GENERATOR_WEIGHTS_PATH = ""
# INPUT_DISCRIMINATOR_WEIGHTS_PATH = ""

INPUT_GENERATOR_WEIGHTS_PATH = "weights/epoch_1_generator.h5"
INPUT_DISCRIMINATOR_WEIGHTS_PATH = "weights/epoch_1_discriminator.h5"                                   # path for input weights (null for training)

# Output Parameters
OUTPUT_IMAGES_PATH = "images"          # path for saved image output (null for no saving)
OUTPUT_WEIGHTS_PATH = "weights"         # path for weight saving (null for no saving)
OUTPUT_IMAGES_COUNT = 3          # number of images to save per epoch
PLOT_LOSS = True                # true to produce a loss plot, if OUTPUT_IMAGE_PATH is non-null saved in the same directory

# Training Variables
EPOCHS = 2      # number of training epochs
BATCH_SIZE = 32

def main():
    dirname = os.path.dirname(__file__)

    # build output directories if they do not yet exist
    if OUTPUT_IMAGES_PATH != "":
        filepath= os.path.join(dirname, OUTPUT_IMAGES_PATH)
        if not os.path.exists(filepath):
            os.mkdir(filepath)

    if OUTPUT_WEIGHTS_PATH != "":
        filepath= os.path.join(dirname, OUTPUT_WEIGHTS_PATH)
        if not os.path.exists(filepath):
            os.mkdir(filepath)

    # create / train model
    style_gan = StyleGAN(epochs=EPOCHS)
    if INPUT_GENERATOR_WEIGHTS_PATH != "" and INPUT_DISCRIMINATOR_WEIGHTS_PATH != "":
        style_gan.built = True

        generator_filepath = os.path.join(dirname, INPUT_GENERATOR_WEIGHTS_PATH)
        discriminator_filepath = os.path.join(dirname, INPUT_DISCRIMINATOR_WEIGHTS_PATH)\

        style_gan.generator.load_weights(generator_filepath)
        style_gan.discriminator.load_weights(discriminator_filepath)
    else:
        style_gan.train(input_images_path=INPUT_IMAGES_PATH,
                        output_images_path=OUTPUT_IMAGES_PATH,
                        images_count=OUTPUT_IMAGES_COUNT, 
                        weights_path=OUTPUT_WEIGHTS_PATH, 
                        plot_loss=PLOT_LOSS) 

    show_example_output(style_gan)

def show_example_output(style_gan):
    z = [tf.random.normal((32, 512)) for i in range(7)]
    noise = [tf.random.uniform([32, res, res, 1]) for res in [4, 8, 16, 32, 64, 128, 256]]
    input = tf.ones([32, 4, 4, 512])

    output_images = style_gan.generator([input, z, noise])
    output_images *= 256
    output_images.numpy()

    img = keras.preprocessing.image.array_to_img(output_images[0])
    img.show(title="Generated Image")

if __name__ == "__main__":
    main()