from GAN import train
from matplotlib import pyplot

### Variables
batch_size = 128
learning_rate_discriminator = 0.0002
learning_rate_generator = 0.0002
generator_input_dim = 8
latent_dim = 256
epochs = 50
data_path = 'D:\\comp3710\\AKOA_Analysis\\'
output_path = 'Resources/'


def plot_history(disc_hist, gen_hist):
	# plot history
	pyplot.plot(disc_hist, label='loss_real')
	pyplot.plot(gen_hist, label='loss_gen')
	pyplot.legend()
	pyplot.savefig('plot_line_plot_loss.png')
	pyplot.close()

def plot_examples(example_images):
    print("Example Dim: ")
    print(example_images.shape)
    f, axarr = pyplot.subplots(2,2)
    axarr[0,0].imshow(example_images[0])
    axarr[0,1].imshow(example_images[1])
    axarr[1,0].imshow(example_images[2])
    axarr[1,1].imshow(example_images[3])
    pyplot.savefig('example_output.png')
    pyplot.close()


def main():
    gen_history, disc_history, example_images = train(data_path, output_path, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim, generator_input_dim=generator_input_dim, learning_rate_generator=learning_rate_generator, learning_rate_discriminator=learning_rate_discriminator)
    plot_history(disc_history, gen_history)
    plot_examples(example_images)


if __name__ == '__main__':
    main()