"""
StyleGAN implementation.

@author Christopher Atkinson
@email c.atkinson@uqconnect.edu.au
"""

import os

# Suppress tensorflow logging:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Set tf memory growth, verify version and cuda/gpu functionality.
print(f'\nTensorflow version: {tf.__version__}')
print(f'Tensorflow CUDA {"is" if tf.test.is_built_with_cuda() else "is not"} available.')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print('Tensorflow set GPU memory growth to True.')
    except RuntimeError as e:
        print(e)
print(f'Tensorflow {"is" if tf.executing_eagerly() else "is not"} executing eagerly.')

import tensorflow_datasets as tfds

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Add, Dense, Flatten, Reshape, LeakyReLU, ReLU, \
    Conv2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import backend as K

import numpy as np
from numpy import average
import itertools
import matplotlib as mpl
from matplotlib import pyplot as plt, gridspec, colors, cm
from matplotlib.ticker import FuncFormatter
from IPython import display
import time

# Colours used in plots.
dark = '#191b26'
darker = '#151722'
cream = '#b3b5be'
mint = '#67a39a'

# Setting matplotlib to dark mode.
mpl.rcParams['text.color'] = cream
mpl.rcParams['axes.labelcolor'] = cream
mpl.rcParams['axes.facecolor'] = dark
mpl.rcParams['axes.edgecolor'] = cream
mpl.rcParams['figure.facecolor'] = darker
mpl.rcParams['xtick.color'] = cream
mpl.rcParams['ytick.color'] = cream


class AdaIN(tf.keras.layers.Layer):
    """
    Apply instanced scale and bias to normalised input.
    """
    def __init__(self, epsilon=1e-5, **kwargs):
        super(AdaIN, self).__init__(**kwargs)
        self.epsilon = epsilon  # To avoid dividing by zero.

    def call(self, inputs, *args, **kwargs):
        """
        Normalises x, then applies individual scale and bias to each channel(feature) of x,
        for every element in the batch.
        """
        x, scale, bias = inputs

        # Reshape (batch*len) y vectors into (batch,1,1,len) to support batched multiplication.
        scale = tf.reshape(scale, shape=(-1, 1, 1, tf.shape(scale)[-1]))
        bias = tf.reshape(bias, shape=(-1, 1, 1, tf.shape(bias)[-1]))

        # Normalise input to be centered on 0 with standard deviation of 1.
        mean = K.mean(x, axis=(1, 2), keepdims=True)
        stddev = K.std(x, axis=(1, 2), keepdims=True) + self.epsilon
        x_norm = (x - mean) / stddev

        return (x_norm * scale) + bias


class StyleGAN:
    """Tensorflow 2 StyleGAN implementation."""

    def __init__(self, dataset=None, dataset_path=None, dataset_name='', target_image_dims=(64, 64),
                 epochs=999, batch_size=32, z_length=100, save_progress_plots=True, show_progress_plots=True,
                 progress_plot_batch_interval=50, save_model_checkpoints=False, model_checkpoint_interval=15,
                 save_directory='./output', print_model_summaries=True, running_in_notebook=False):
        """
        Initialise generator and discriminator, and read in dataset.

        :param dataset: (Optional) pass in your own dataset.
        :param dataset_path: Path to local dataset. Path should lead to a folder of images.
        :param dataset_name: Key words to describe the dataset (eg. 'oasis', 'knee', 'mnist').
        :param target_image_dims: The desired dimensions to reshape the input to and shape the output to.
        Dimensions must be square and must not exceed the input data dimensions.
        :param epochs: Number of epochs to train for.
        :param batch_size: Number of samples per batch. Lower this if training fails due to OOM.
        :param z_length: Length of the latent vector z. 100 is sufficient for MNIST Digits, but larger datasets
        require ~512 to avoid mode collapse.
        :param save_progress_plots: Whether to export progress plots.
        :param show_progress_plots: Whether to show progress plots (only set to true if running in jupyter notebook)
        :param progress_plot_batch_interval: How often to generate a new progress plot.
        :param save_model_checkpoints: Whether or not to save model checkpoints.
        :param model_checkpoint_interval: How many epochs to have elapsed before saving a new model checkpoint.
        :param save_directory: Path to save progress plots and checkpoints to.
        :param print_model_summaries: Whether to print summaries of models upon creation.
        :param running_in_notebook: Whether this is being executed in a jupyter notebook.
        """
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.z_length = z_length
        self.jupyter = running_in_notebook
        self.save_progress_plots = save_progress_plots
        self.show_progress_plots = show_progress_plots
        self.progress_interval = progress_plot_batch_interval
        self.save_model_checkpoints = save_model_checkpoints
        self.model_checkpoint_interval = model_checkpoint_interval

        # Set up output directories.
        self.save_directory = save_directory
        self.plot_directory = save_directory + '/progress_plots'
        self.checkpoint_directory = save_directory + '/model_checkpoints'
        for path in (self.save_directory, self.plot_directory, self.checkpoint_directory):
            if not os.path.exists(path) and os.access(os.path.dirname(path), os.W_OK):
                os.makedirs(path, exist_ok=True)
                print(f'Created folder: {path}')

        self.max_graphed = 50  # Number of loss values to use in progress plot.
        self.colorbar_norm = mpl.colors.TwoSlopeNorm(vmin=-1.05, vcenter=-0.1, vmax=1.05)

        # Whether to use MNIST dataset from tfds or read in dataset from provided path.
        if dataset is None and dataset_path:
            self.dataset = self.get_local_image_dataset(dataset_path, target_image_dims)
        elif any(name in dataset_name for name in ('mnist', 'digit')):
            self.dataset = self.get_mnist_dataset()
        self.check_dataset()

        # Determine image dimensions
        image = list(self.dataset.take(1).as_numpy_iterator())[0][0]
        self.image_size = image.shape[0]
        self.num_channels = image.shape[-1]

        self.start_size, self.num_blocks = self._smallest_by_halving()

        self.params = self.get_hyperparameters(dataset_name)

        self.generator = self.make_generator()
        self.discriminator = self.make_discriminator()

        # Determine number of trainable weights in each model.
        self.num_gen_weights = int(np.sum([np.prod(v.get_shape()) for v in self.generator.trainable_weights]))
        self.num_disc_weights = int(np.sum([np.prod(v.get_shape()) for v in self.discriminator.trainable_weights]))

        self.gen_optimizer = Adam(learning_rate=self.params['learning_rate_gen'], beta_1=self.params['beta_1'])
        self.disc_optimizer = Adam(learning_rate=self.params['learning_rate_disc'], beta_1=self.params['beta_1'])

        self.checkpoint_prefix = os.path.join(self.checkpoint_directory, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer,
                                              discriminator_optimizer=self.disc_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

        # Common seed comprising input z and noise, to be used repeatedly within progress plots.
        self.progress_seed = self.generate_inputs(batch_size=8)

        self.g_losses, self.d_real_losses, self.d_fake_losses = [], [], []
        self.g_loss_avg, self.d_fake_loss_avg, self.d_real_loss_avg, self.d_loss_avg = 0, 0, 0, 0
        self.balance = 0  # Adversarial balance.

        self.start_time = 0
        self.iteration = 0
        self.num_batches = len(self.dataset) - 1
        self.current_batch = 0
        self.current_epoch = 0

        # Determine variance of dataset (within shuffled sample).
        self.dataset_variance, self.dataset_variance_sum = self._calculate_variance(
            list(self.dataset.take(1).as_numpy_iterator())[0])

        if print_model_summaries:
            print()
            self.discriminator.summary()
            print()
            self.generator.summary()
            print()
        self.output_model_plots()

    def get_hyperparameters(self, dataset_name):
        """
        Retrieves the optimised hyperparameters for the named dataset.
        :param dataset_name: Key words describing dataset.
        """
        dataset_name = dataset_name.lower()
        print()
        if any(name in dataset_name for name in ('mnist', 'digits')):
            print('MNIST Digits hyperparameter presets loaded.')
            # Tested at (28,28,1) image size.
            params = {
                'learning_rate_gen': 0.0002,
                'learning_rate_disc': 0.0002,
                'beta_1': 0.5,
                'gen_filters': 32,
                'disc_filters': 16,
            }
        elif any(name in dataset_name for name in ('celeba', 'faces')):
            print('CelebA Faces hyperparameter presets loaded.')
            # Tested at (64,64,1) image size.
            params = {
                'learning_rate_gen': 0.00005,
                'learning_rate_disc': 0.00005,
                'beta_1': 0.5,
                'gen_filters': 128,
                'disc_filters': 64,
            }
        elif any(name in dataset_name for name in ('brains', 'oasis')):
            print('OASIS Brains hyperparameter presets loaded.')
            # Tested at (64,64,1) image size.
            params = {
                'learning_rate_gen': 0.00002,
                'learning_rate_disc': 0.00001,
                'beta_1': 0.5,
                'gen_filters': 256,
                'disc_filters': 256,
            }
        elif any(name in dataset_name for name in ('oai', 'akoa', 'knees')):
            print('OAI AKOA Knee hyperparameter presets loaded.')
            # Tested at (64, 64, 1) image size.
            params = {
                'learning_rate_gen': 0.00002,
                'learning_rate_disc': 0.00001,
                'beta_1': 0.5,
                'gen_filters': 128,
                'disc_filters': 64,
            }
        else:  # Default
            print('WARNING: Default hyperparameter presets loaded. These may not work well with your dataset.')
            params = {
                'learning_rate_gen': 0.0002,
                'learning_rate_disc': 0.0002,
                'beta_1': 0.5,
                'gen_filters': 64,
                'disc_filters': 32,
            }
        return params

    def get_mnist_dataset(self):
        """Retrieve MNIST Digits dataset from tfds. Normalise, shuffle and batch the dataset."""
        dataloader = tfds.load('mnist', as_supervised=True)
        dataset = dataloader['train']
        # Cast [0,255] images to [-1,1].
        dataset = dataset.map(lambda image, label: Rescaling(scale=1. / 127.5, offset=-1)(image))
        dataset = dataset.shuffle(self.batch_size).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        print('MNIST Digits dataset loaded.')
        return dataset

    def get_local_image_dataset(self, path, image_dims, make_grayscale=True):
        """
        Retrieve dataset from provided path, before rescaling pixel values to [-1,1] and converting to grayscale.

        :param path: Path to dataset (must be a folder containing only images).
        :param image_dims: Dimensions to scale images to.
        :param make_grayscale: Whether or not to convert images to grayscale.
        :return:
        """
        print(f'Loading dataset from {path} at {image_dims} resolution.')
        dataset = tf.keras.preprocessing.image_dataset_from_directory(path,
                                                                      labels=None,
                                                                      label_mode=None,
                                                                      image_size=image_dims,
                                                                      smart_resize=True,
                                                                      shuffle=True,
                                                                      batch_size=self.batch_size)
        # Rescale all [0,255] images to [-1,1], as our generator outputs with tanh. Also convert to 1-channel.
        dataset = dataset.map(lambda x: Rescaling(scale=1. / 127.5, offset=-1)(
            tf.image.rgb_to_grayscale(x) if make_grayscale else x))
        dataset = dataset.shuffle(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def check_dataset(self):
        """Retrieve and plot one sample from the active dataset."""
        print('Sample from dataset:')
        batch = self.dataset.take(1)
        image = list(batch.as_numpy_iterator())[0][0]
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(image, cmap='gray')
        plt.show()

    def _calculate_variance(self, images, epsilon=1e-7):
        """
        Calculate pixel-wise variance among all permutations of image pairs.

        :param images: Array of images to use.
        :param epsilon: Small value to avoid dividing by zero.
        :return: pixel-wise variance, and sum of variance.
        """
        diff, count = np.zeros(shape=images[0].shape), 0
        for x in itertools.combinations(images, 2):  # Compute all permutations of pairs.
            diff = diff + np.abs(x[0] - x[1]) + epsilon
            count += 1
            if count > 500:  # Short-circuit if the sample is too large.
                break
        diff = diff / count
        return diff, np.sum(diff)

    def output_model_plots(self):
        """Write image of model architectures to directory."""
        tf.keras.utils.plot_model(self.generator, show_shapes=True,
                                  to_file=self.save_directory + '/generator_plot.png')
        tf.keras.utils.plot_model(self.discriminator, show_shapes=True,
                                  to_file=self.save_directory + '/discriminator_plot.png')
        print(f'Model architecture plots saved to {self.save_directory}/\n')

    def _smallest_by_halving(self):
        """
        Get smallest round integer by halving the input repeatedly.
        This becomes our starting generator convolution size.
        Also return number of halves performed.
        This becomes our number of generator and discriminator blocks, to
        return us from the reduced size to the original resolution.
        """
        x, count = self.image_size, 0
        while (x / 2) % 1 == 0 and (x / 2) > 2:
            x = x / 2
            count += 1
        if x >= (self.image_size / 2):
            raise ValueError(f'{self.image_size} may be unsuitable for upsampling, '
                             f'as its smallest common component is {x}, '
                             f'which is >= input of {self.image_size}.')
        return int(x), count

    def _disc_block(self, x, size, i, reduce=True):
        """
        Modular discriminator block
        :param x: Input matrix.
        :param size: Number of filters.
        :param i: Which numbered block this is.
        :param reduce: Whether to apply pooling.
        """
        x = Conv2D(filters=size, kernel_size=3, strides=1, padding='same',
                   kernel_initializer=GlorotNormal(), activation=None,
                   name=f'conv{i}')(x)
        if reduce:
            x = AveragePooling2D(name=f'avg_pool{i}')(x)
        return LeakyReLU(0.2)(x)

    def _gen_block(self, x, w, noise, size, i):
        """
        Modular generator block.
        :param x: Input matrix.
        :param w: Latent w vector.
        :param noise: Gaussian noise matrix.
        :param size: Number of neurons/filters.
        :param i: Which numbered block this is.
        """
        # Affine transformation A:
        scale = Dense(size, name=f'scale{i}')(w)
        bias = Dense(size, name=f'bias{i}')(w)
        # Transformation B:
        noise = Dense(size, name=f'noise{i}')(noise)

        x = UpSampling2D(name=f'upscale{i}')(x)
        x = Conv2D(filters=size, kernel_size=3, strides=1, padding='same',
                   kernel_initializer=GlorotNormal(), use_bias=False,
                   activation=None, name=f'conv{i}')(x)
        x = Add(name=f'add_noise{i}')([x, noise])
        x = AdaIN(name=f'ada_in{i}')([x, scale, bias])
        return ReLU(name=f'relu{i}')(x)

    def make_generator(self):
        """Construct generator (synthesis network)"""
        # Process all inputs - const vector, z vectors and noise matrices.
        const = Input(shape=(self.z_length,), name='const')
        z, noise, size = [], [], self.start_size
        for i in range(self.num_blocks):
            z.append(Input(shape=(self.z_length,), name=f'z{i}'))
            noise.append(Input(shape=(size * 2, size * 2, self.num_channels), name=f'noise_in{i}'))
            size *= 2

        # Mapping network
        latents = Input(shape=(self.z_length,), name='z_input')
        w = Dense(64, activation=LeakyReLU(0.2), name='w0')(latents)
        for i in range(6):
            w = Dense(64, activation=LeakyReLU(0.2), name=f'w{i + 1}')(w)
        w = Dense(self.params['gen_filters'], activation=LeakyReLU(0.2), name=f'w7')(w)
        map = Model(inputs=latents, outputs=w, name='mapping_network')

        # Start block
        x = Dense(self.start_size * self.start_size * self.params['gen_filters'],
                  use_bias=True, activation='relu', kernel_initializer=GlorotNormal(),
                  name='const_expander')(const)
        x = Reshape([self.start_size, self.start_size, self.params['gen_filters']], name='const_reshape')(x)

        # Generator blocks
        for i in range(self.num_blocks):
            w = map(z[i])
            x = self._gen_block(x, w, noise[i], self.params['gen_filters'], i)

        # Convert to n-channel image with values bounded by tanh.
        image = Conv2D(filters=1, kernel_size=1, strides=1, padding='same',
                       kernel_initializer=GlorotNormal(), activation='tanh', name='image_output')(x)

        model = Model(inputs=[const] + z + noise, outputs=[image], name='generator')
        print('Generator model constructed.')
        return model

    def make_discriminator(self):
        """Construct discriminator."""
        image = Input([self.image_size, self.image_size, self.num_channels], name='image_in')

        # Start block
        x = self._disc_block(image, self.params['disc_filters'], 'start', reduce=False)

        # Discriminator blocks
        for i in range(self.num_blocks):
            x = self._disc_block(x, self.params['disc_filters'], i, reduce=True)

        # Output block - was the image fake or real?
        x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                   kernel_initializer=GlorotNormal(), activation=LeakyReLU(0.2), name='conv_final')(x)
        x = Flatten(name='flatten_conv')(x)
        classification = Dense(1, activation='sigmoid', name='classification_out')(x)

        model = Model(inputs=image, outputs=classification, name='discriminator')
        print('Discriminator model constructed.')
        return model

    def generate_inputs(self, batch_size):
        """
        Randomly generated latent z vectors and noise matrices to use as inputs to the generator.
        :param batch_size: Number of samples to return.
        """
        const = tf.random.normal(mean=0., stddev=1., shape=(batch_size, self.z_length))
        z, noise, size = [], [], self.start_size
        for i in range(self.num_blocks):
            # Latent z vectors.
            z.append(tf.random.normal(mean=0., stddev=1., shape=(batch_size, self.z_length)))
            # Random noise matrices.
            noise.append(
                tf.random.normal(mean=0., stddev=1., shape=(batch_size, size * 2, size * 2, self.num_channels)))
            # Double the size of noise matrices each iteration, as each block outputs an up-scaled image.
            size *= 2
        return [const] + z + noise

    def _update_avg_losses(self):
        """Compute weighted moving average of loss values, to show adversarial balance in colorbar."""
        if len(self.g_losses) > 10:
            weights = np.arange(1, 11)
            self.g_loss_avg = round(average(self.g_losses[len(self.d_fake_losses) - 10:], weights=weights), 3)
            self.d_fake_loss_avg = round(average(self.d_fake_losses[len(self.d_fake_losses) - 10:], weights=weights), 3)
            self.d_real_loss_avg = round(average(self.d_real_losses[len(self.d_real_losses) - 10:], weights=weights), 3)
        elif len(self.g_losses) > 0:
            weights = np.arange(1, len(self.g_losses) + 1)
            self.g_loss_avg = round(average(self.g_losses, weights=weights), 3)
            self.d_fake_loss_avg = round(average(self.d_fake_losses, weights=weights), 3)
            self.d_real_loss_avg = round(average(self.d_real_losses, weights=weights), 3)
        self.d_loss_avg = (self.d_fake_loss_avg + self.d_real_loss_avg) / 2
        # Balance is the difference between the weighted average of generator and discriminator losses.
        # Negative balance implies a weak discriminator, while positive balance implies a weak generator.
        # Ideal balance is 0.
        self.balance = self.g_loss_avg - self.d_loss_avg
        # Bound the value between [-1,1] so we don't draw off the edge of the bar.
        if self.balance < -1:
            self.balance = -1
        elif self.balance > 1:
            self.balance = 1

    def _plot_gan_progress(self):
        """
        Plot sample generated images over losses, balance and variance during training.
        """
        if not self.jupyter:
            # Matplotlib does not automatically close plots outside of jupyter.
            plt.close('all')

        self._update_avg_losses()

        # Generate generator sample images.
        gen_images = self.generator(self.progress_seed, training=False)

        # Calculate pixel-wise variance among generated samples.
        sample_variance, var_sum = self._calculate_variance(gen_images)
        # Convert sum of variance to a percentage of dataset variance.
        var_percent = (var_sum / self.dataset_variance_sum) * 100

        fig = plt.figure(constrained_layout=False, figsize=(14.5, 11))
        fig.suptitle(f'Trainable parameters of Generator: {self.num_gen_weights:,}, '
                     f'Discriminator: {self.num_disc_weights:,}\n'
                     f'Epoch {self.current_epoch:03} / {self.epochs}  |  '
                     f'Batch {self.current_batch:03} / {self.num_batches}  |  '
                     f'Time elapsed: {round(((time.time() - self.start_time) / 60) / 60, 3):.3f} hours',
                     size=15, linespacing=1.6)

        # Parent container gridspec.
        gs0 = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[2, 1], width_ratios=[1], figure=fig,
                                left=0.045, right=0.955, top=.925, bottom=0.035, hspace=0.1)
        # Top gridspec (images).
        gs1 = gs0[0, 0].subgridspec(ncols=4, nrows=2, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1],
                                    wspace=.05, hspace=.05)
        # Bottom gridspec (losses and balance).
        gs2 = gs0[1, 0].subgridspec(ncols=3, nrows=1, height_ratios=[1], width_ratios=[5, 0.7, 2],
                                    wspace=.1, hspace=0)

        # Create sample image subplots.
        ax = []
        for i in range(2):
            for j in range(4):
                ax.append(fig.add_subplot(gs1[i, j]))

        # Create subplots for losses, balance and variance.
        ax8 = fig.add_subplot(gs2[0, 0])  # Losses
        ax9 = fig.add_subplot(gs2[0, 1])  # Balance
        ax10 = fig.add_subplot(gs2[0, 2])  # Variance

        # Plot Generator sample images.
        for i in range(gen_images.shape[0]):
            ax[i].imshow(gen_images[i].numpy(), cmap='gray')
            ax[i].axes.xaxis.set_visible(False)
            ax[i].axes.yaxis.set_visible(False)

        # Plot losses of generator and discriminator.
        ax8.set_title('StyleGAN Generator and Discriminator binary cross-entropy losses over time', size=14)
        ax8.set_xticklabels([])
        # Truncate loss lists to only keep n latest values.
        if len(self.g_losses) >= self.max_graphed:
            self.g_losses = self.g_losses[len(self.g_losses) - self.max_graphed:]
            self.d_real_losses = self.d_real_losses[len(self.d_real_losses) - self.max_graphed:]
            self.d_fake_losses = self.d_fake_losses[len(self.d_fake_losses) - self.max_graphed:]
        ax8.plot(self.g_losses, c=mint, label=f'Generator loss')
        ax8.plot(self.d_real_losses, c='purple', label=f'Discriminator loss vs real')
        ax8.plot(self.d_fake_losses, c='#ff884d', label=f'Discriminator loss vs fake')
        # Put loss value at tail end of each loss line.
        ax8.text(x=len(self.g_losses) - 1, y=self.g_losses[-1], s=f'{round(self.g_losses[-1], 3):.3f}')
        ax8.text(x=len(self.d_real_losses) - 1, y=self.d_real_losses[-1], s=f'{round(self.d_real_losses[-1], 3):.3f}')
        ax8.text(x=len(self.d_fake_losses) - 1, y=self.d_fake_losses[-1], s=f'{round(self.d_fake_losses[-1], 3):.3f}')
        # Allow room for text at the end.
        ax8.set_xlim((0, int(self.max_graphed + (self.max_graphed * 0.04))))
        # Force y axis tick labels to have exactly two decimals, to stop graph width jumping.
        ax8.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f'{x:0.2f}'))
        # Force 1 tick mark per value.
        ax8.xaxis.set_major_locator(plt.MaxNLocator(int(self.max_graphed + (self.max_graphed * 0.04))))
        # Show background grid. Draws lines from each tick label.
        ax8.grid(color='gray', alpha=0.2)
        # Move legend to the left after lines reach the halfway point.
        ax8.legend(loc='upper right' if len(self.g_losses) < self.max_graphed / 2 else 'upper left', fontsize=13)

        # Plot adversarial balance colorbar.
        ax9.set_title('Weak Generator', size=11)
        ax9.set_xlabel('Weak Discriminator', size=11)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=self.colorbar_norm, cmap='rainbow'), cax=ax9)
        ax9.set_yticklabels([])
        cb.ax.yaxis.set_ticks_position('none')
        cb.ax.set_ylabel('Adversarial Balance', fontsize=12, labelpad=2)
        cb.ax.yaxis.set_label_position('left')
        # Plot acceptable midpoints.
        cb.ax.axhline(y=0.15, c='gray', linestyle='--', linewidth=1)
        cb.ax.axhline(y=-0.15, c='gray', linestyle='--', linewidth=1)
        # Plot actual balance line.
        cb.ax.axhline(y=self.balance, c='black', linestyle='-', linewidth=7)
        cb.ax.axhline(y=self.balance, c='lime' if -0.15 < self.balance < 0.15 else 'red', linestyle='-', linewidth=5)

        # Plot generator variance.
        ax10.set_title(f'Generator variance: {int(round(var_percent)):03}%', size=14)
        ax10.imshow(sample_variance, cmap='viridis')
        ax10.axes.xaxis.set_visible(False)
        ax10.axes.yaxis.set_visible(False)

        self.iteration += 1
        if self.save_progress_plots:
            plt.savefig(f'{self.plot_directory}/step_{self.iteration:08}.png')
        if self.show_progress_plots:
            plt.show()

    def _disc_loss(self, real_output, fake_output):
        """
        How well the discriminator can distinguish real from fake images.
        :param real_output: Classifications of real images.
        :param fake_output: Classifications of fake images.
        :return: Loss on real images, loss on fake images.
        """
        # Compare predictions on real images to array of ones.
        real_loss = BinaryCrossentropy(label_smoothing=0.2)(tf.ones_like(real_output), real_output)
        # Compare predictions on fake images to array of zeroes.
        fake_loss = BinaryCrossentropy(label_smoothing=0.2)(tf.zeros_like(fake_output), fake_output)

        return real_loss, fake_loss

    def _gen_loss(self, fake_output):
        """
        How well the generator can fool the discriminator.
        We get this loss from the discriminator, and pass it to the generator.
        :param fake_output: Discriminator classification of fake images.
        :return: Loss of discriminator on fake images, inverted.
        """
        # Compare discriminator decisions to array of ones.
        return BinaryCrossentropy()(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        """
        Conduct forward and backward pass, updating weights of each model
        by their loss.
        :param images: Batch of images (either real or fake)
        :return: Losses, for plotting only.
        """
        # Random latent space input for generator.
        inputs = self.generate_inputs(batch_size=self.batch_size)

        # Track gradients of each model.
        # ie. Track what happened in what order during forward pass.
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Forward pass.
            generated_images = self.generator(inputs, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            g_loss = self._gen_loss(fake_output)
            d_real_loss, d_fake_loss = self._disc_loss(real_output, fake_output)
            d_loss = d_real_loss + d_fake_loss

        # Backward pass.
        # Calculate gradient for each models trainable weights.
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)

        # Update generator and discriminator weights with gradients.
        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return g_loss, d_real_loss, d_fake_loss

    def train(self):
        """Training loop."""
        self.start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self.current_epoch = epoch

            for i, image_batch in enumerate(self.dataset):
                self.current_batch = i

                # Start training step, tracking losses.
                g_loss, d_real_loss, d_fake_loss = self.train_step(image_batch)
                g_loss, d_real_loss, d_fake_loss = g_loss.numpy(), d_real_loss.numpy(), d_fake_loss.numpy()

                # Update output graph every n batches.
                if (self.save_progress_plots or self.show_progress_plots) and i % self.progress_interval == 0:
                    self.g_losses.append(g_loss)
                    self.d_real_losses.append(d_real_loss)
                    self.d_fake_losses.append(d_fake_loss)
                    if self.jupyter:
                        display.clear_output(wait=True)
                    self._plot_gan_progress()
                if not self.jupyter or not self.show_progress_plots:
                    # Print training progress.
                    print(
                        f'\rEpoch {epoch:03} / {self.epochs} | batch {i:03} / {self.num_batches} | '
                        f'g_loss: {round(g_loss, 4):.4f} | d_fake_loss: {round(d_fake_loss, 4):.4f} | '
                        f'd_real_loss: {round(d_real_loss, 4):.4f} | '
                        f'Time taken: {round(((time.time() - epoch_start_time) / 60), 2):.2f} minutes', end='')
            print()

            # Save model checkpoint every n epochs.
            if self.save_model_checkpoints and epoch % self.model_checkpoint_interval == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)


if __name__ == '__main__':
    # This is used for testing. The driver script is where you should execute training.
    gan = StyleGAN(dataset=None,

                   # dataset_path=None,
                   # dataset_name='mnist digits',

                   # dataset_path='C:/img_align_celeba',
                   # dataset_name='celeb faces',

                   dataset_path='C:/AKOA_Analysis',
                   dataset_name='oai akoa knees',

                   # dataset_path='C:/OASIS_brains/keras_png_slices_train',
                   # dataset_name='oasis brains',

                   target_image_dims=(64, 64),
                   epochs=999,
                   batch_size=32,
                   z_length=512,
                   save_progress_plots=True,
                   show_progress_plots=False,
                   progress_plot_batch_interval=10,
                   save_model_checkpoints=False,
                   model_checkpoint_interval=20,
                   save_directory='C:/stylegan_output',
                   print_model_summaries=True,
                   running_in_notebook=False)

    gan.train()
