import tensorflow as tf
from PIL import Image
import glob
import numpy as np
from matplotlib import pyplot

def load_data(filepath, batch_size):
    image_files = glob.glob(filepath + '*')
    images = np.array([np.array(Image.open(i).resize((128,128))) for i in image_files])

    discriminator_input_dim = images.shape[1:]
    dataset_size = images.shape[0]

    images = images/255

    print("Data Shape:")
    print(images.shape)

    images = tf.data.Dataset.from_tensor_slices(images)
    images = images.shuffle(buffer_size=batch_size)
    images = images.repeat().batch(batch_size)
    image_iter = iter(images)

    return image_iter, discriminator_input_dim, dataset_size

def load_test_data(filepath):
    image_files = glob.glob(filepath + '*')
    images = np.array([np.array(Image.open(i).resize((128,128))) for i in image_files])

    dataset_size = images.shape[0]

    print("Test Data Shape:")
    print(images.shape)

    images = tf.data.Dataset.from_tensor_slices(images)
    image_iter = iter(images)

    return image_iter, dataset_size


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
    dim = 4
    f, axarr = pyplot.subplots(dim,dim)
    for i in range(dim):
        for j in range(dim):
            axarr[i,j].imshow(example_images[dim * i + j])
    pyplot.savefig('example_output.png')
    pyplot.close()


def ssim(test_filepath, example_images):
    test_dataset, dataset_size = load_test_data(test_filepath)

    iterations = min(dataset_size, example_images.shape[0])

    ssims = []

    for i in range(iterations):
        ssims.append(tf.image.ssim(test_dataset.next(), example_images[i]))

    ssims = np.ndarray(ssims)

    return np.mean(ssims)

def generate_example_images(gen, num_examples):
    latent_data = tf.random.normal(shape=(num_examples, latent_dim))
    fake_images = gen(latent_data).numpy() * 255
    mins = tf.math.reduce_min(fake_images, axis=(1,2,3))[:,None,None,None]
    maxs = tf.math.reduce_max(fake_images, axis=(1,2,3))[:,None,None,None]
    fake_images = (fake_images - mins)/(maxs-mins)
    return fake_images