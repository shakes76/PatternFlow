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
    images = images.repeat()
    image_iter = iter(images)

    return image_iter, dataset_size


def plot_history(disc_hist, gen_hist, output_path):
	# plot history
	pyplot.plot(disc_hist, label='loss_disc')
	pyplot.plot(gen_hist, label='loss_gen')
	pyplot.legend()
	pyplot.savefig(output_path + 'plot_line_plot_loss.png')
	pyplot.close()

def plot_examples(example_images, output_path):
    print("Example Dim: ")
    print(example_images.shape)
    dim = 4
    f, axarr = pyplot.subplots(dim,dim)
    for i in range(dim):
        for j in range(dim):
            axarr[i,j].imshow(example_images[dim * i + j])
    pyplot.savefig(output_path + 'example_output.png')
    pyplot.close()


def calculate_ssim(test_filepath, example_images):
    test_dataset, dataset_size = load_test_data(test_filepath)

    #iterations = min(dataset_size, example_images.shape[0])

    ssims = []

    

    for i in range(example_images.shape[0]):
        for j in range(dataset_size):
            ssims.append(tf.image.ssim(test_dataset.get_next(), example_images[i], max_val=255))

    ssims = np.asarray(ssims)
    real = test_dataset.get_next()

    tf.keras.preprocessing.image.save_img('real.png', real)
    tf.keras.preprocessing.image.save_img('fake.png', example_images[0])


    return np.mean(ssims)

def generate_example_images(gen, num_examples, latent_dim):
    latent_data = tf.random.normal(shape=(num_examples, latent_dim))
    fake_images = gen(latent_data)
    mins = tf.math.reduce_min(fake_images, axis=(1,2,3))[:,None,None,None]
    maxs = tf.math.reduce_max(fake_images, axis=(1,2,3))[:,None,None,None]
    fake_images = (fake_images - mins)/(maxs-mins)
    fake_images = fake_images*255
    fake_images = tf.cast(fake_images, dtype=tf.uint8)
    return fake_images