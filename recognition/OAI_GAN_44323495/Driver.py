'''
Driver script for running the GAN

This script demonstrates the process for training the GAN on the OAI AKOA Knee dataset.

Requirements:
- GAN
- DataUtils

Author: Erik Brand
Date: 01/11/2020
License: Open Source
'''

from GAN import train
from DataUtils import *

### Variables
batch_size = 128
learning_rate_discriminator = 0.0002
learning_rate_generator = 0.0002
generator_input_dim = 8
latent_dim = 256
epochs = 35
data_path = 'D:/comp3710/AKOA_Analysis/'
test_data_path = 'D:/comp3710/AKOA_Analysis_test/'
output_path = 'Resources/'

# Main Driver Function
def main():
    # Train the model
    gen_history, disc_history, gen = train(data_path, output_path, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim, generator_input_dim=generator_input_dim, learning_rate_generator=learning_rate_generator, learning_rate_discriminator=learning_rate_discriminator, debug=False)
    
    # Construct Output
    plot_history(disc_history, gen_history, output_path)

    example_images = generate_example_images(gen, 16, latent_dim)
    plot_examples(example_images, output_path)

    ssim = calculate_ssim(test_data_path, example_images, output_path)
    print("SSIM: " + str(ssim))



if __name__ == '__main__':
    main()