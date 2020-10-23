from GAN import train
from matplotlib import pyplot

### Variables
batch_size = 128
learning_rate_discriminator = 0.0002
learning_rate_generator = 0.0002
generator_input_dim = 16
latent_dim = 256
epochs = 30
data_path = 'D:\\comp3710\\AKOA_Analysis\\'
output_path = 'Resources/'


gen_history, disc_history = train(data_path, output_path, generator_input_dim=generator_input_dim)