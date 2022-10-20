import tensorflow as tf
import os
from tensorflow import keras
from train import StyleGAN

#=======INPUT=FILE=LOCATIONS=========

# path for input images (non-null unless using a pretrained model)
INPUT_IMAGES_PATH = "keras_png_slices_data"      

# path for input weights ("" for training)
INPUT_GENERATOR_WEIGHTS_PATH = ""           
INPUT_DISCRIMINATOR_WEIGHTS_PATH = ""

# INPUT_GENERATOR_WEIGHTS_PATH = "weights/epoch_1_generator.h5"
# INPUT_DISCRIMINATOR_WEIGHTS_PATH = "weights/epoch_1_discriminator.h5"

#=======OUTPUT=FILE=LOCATIONS=========

# path for saved image output ("" for no saving)
OUTPUT_IMAGES_PATH = "images"

# path for saved weight output ("" for no saving)
OUTPUT_WEIGHTS_PATH = "weights"

# number of output images per epoch
OUTPUT_IMAGES_COUNT = 3

# true to produce a loss plot, saved in the same directory as OUTPUT_IMAGES_PATH
PLOT_LOSS = True                

#=======TRAINING=VARIABLES=========

# number of training epochs
EPOCHS = 100

# training batch size
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
    style_gan = StyleGAN(epochs=EPOCHS, batch_size=BATCH_SIZE)
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
    generator_inputs = style_gan.get_generator_inputs()

    output_images = style_gan.generator(generator_inputs)
    output_images *= 256
    output_images.numpy()

    img = keras.preprocessing.image.array_to_img(output_images[0])
    img.show(title="Generated Image")

if __name__ == "__main__":
    main()