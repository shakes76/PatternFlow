from pathlib import Path
from utils import Config, plot, preview_images, hms_string
from tensorflow.keras import initializers, layers, metrics, \
                             models, optimizers, losses
import logging
import tensorflow as tf
import time

log = logging.getLogger(__name__)

class GAN:
    '''
    A DCGAN for 128x128 images
    '''
    def __init__(self, config: Config):
        self.config = config
        self.strategy = tf.distribute.get_strategy()

        self.generator_optimizer = optimizers.Adam(
            config.generator_lr, config.generator_beta1)
        self.discriminator_optimizer = optimizers.Adam(
            config.discriminator_lr, config.discriminator_beta1)

        self.generator = build_generator(config)
        self.discriminator = build_discriminator(config)

        self.loss_object = losses.BinaryCrossentropy(
            from_logits=True,
            reduction=losses.Reduction.NONE
        )

        self.checkpoint_prefix = config.output_dir / 'checkpoints' / 'ckpt'
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

        self.loss_gs = []
        self.loss_ds = []
        self.d_reals = []
        self.d_fakes = []
        self.images = []

    def discriminator_loss(self, real_output, fake_output):
        '''
        (Distributed) Loss function for the discriminator
        '''
        loss_object = self.loss_object
        global_batch_size = self.config.global_batch_size

        # Per example loss
        real_loss = loss_object(tf.ones_like(real_output), real_output)
        fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)

        # Compute the average loss scaled by global batch size
        real_average_loss = tf.nn.compute_average_loss(real_loss, global_batch_size=global_batch_size)
        fake_average_loss = tf.nn.compute_average_loss(fake_loss, global_batch_size=global_batch_size)
        total_loss = real_average_loss + fake_average_loss
        return total_loss

    def generator_loss(self, fake_output):
        '''
        (Distributed) Loss function for the generator
        '''
        loss_object = self.loss_object
        global_batch_size = self.config.global_batch_size
        
        # Per example loss
        loss = loss_object(tf.ones_like(fake_output), fake_output)

        # Compute the average loss scaled by global batch size
        average_loss = tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)
        return average_loss

    @tf.function
    def distributed_epoch_step(self, images):
        '''
        A step in the epoch. This function takes the variables from all copies
        of the model (replicas) and reduces it to a summary value.
        
        See: https://www.tensorflow.org/tutorials/distribute/keras#overview
        '''
        strategy = self.strategy

        loss_g, loss_d, d_real, d_fake = strategy.run(self.epoch_step, args=(images, ))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, loss_g, axis=None), \
                strategy.reduce(tf.distribute.ReduceOp.SUM, loss_d, axis=None), \
                strategy.reduce(tf.distribute.ReduceOp.MEAN, d_real, axis=None), \
                strategy.reduce(tf.distribute.ReduceOp.MEAN, d_fake, axis=None)

    def epoch_step(self, images):
        '''A step in the epoch.'''
        generator = self.generator
        discriminator = self.discriminator
        generator_optimizer = self.generator_optimizer
        discriminator_optimizer = self.discriminator_optimizer

        seed = tf.random.normal([self.config.batch_size, self.config.seed_size])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(seed, training=True)
            d_real = discriminator(images, training=True)
            d_fake = discriminator(generated_images, training=True)

            loss_g = self.generator_loss(d_fake)
            loss_d = self.discriminator_loss(d_real, d_fake)

        gradients_of_generator = gen_tape.gradient(loss_g, 
            generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(loss_d, 
            discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, 
            generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, 
            discriminator.trainable_variables))

        return loss_g, loss_d, tf.reduce_mean(d_real), tf.reduce_mean(d_fake)

    def epoch(self, images, epoch: int):
        '''An epoch'''
        start = time.time()
        
        total_loss_g, total_loss_d, total_d_real, total_d_fake = 0, 0, 0, 0
        batches = 0
        for batch in images:
            loss_g, loss_d, d_real, d_fake = self.distributed_epoch_step(batch)
            total_loss_g += loss_g
            total_loss_d += loss_d
            total_d_real += d_real
            total_d_fake += d_fake
            batches += 1

        loss_g = total_loss_g/batches
        loss_d = total_loss_d/batches
        d_real = total_d_real/batches
        d_fake = total_d_fake/batches

        elapsed = time.time() - start

        log.info(f'Epoch {epoch}, loss_g={loss_g:5f}, loss_d={loss_d:5f} '
            f'd_real={d_real:5f}, d_fake={d_fake:5f}, {hms_string(elapsed)}')

        self.loss_gs.append(loss_g)
        self.loss_ds.append(loss_d)
        self.d_reals.append(d_real)
        self.d_fakes.append(d_fake)

        # checkpoint
        if (epoch + 1) % self.config.checkpoint_freq == 0:
            log.info('Saving checkpoint...')
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        
        #preview = self.generator.predict(self.config.fixed_seed)
        #self.images.append(preview_images(self.config, preview, epoch))

        return loss_g, loss_d, d_real, d_fake

    def train(self, images, epochs):
        '''Train the DCGAN'''
        start = time.time()

        for i in range(epochs):
            loss_g, loss_d, d_real, d_fake = self.epoch(images, i)

            #output_path = self.config.output_dir / 'output'
            #output_path.mkdir(exist_ok=True)
            #filename = output_path / f"train-{i}.png"
            #self.images[-1].save(filename)
            #filename = output_path / f"train.png"
            #self.images[-1].save(filename)

            #plot(self.loss_gs, self.loss_ds, self.d_reals, 
            #    self.d_fakes, self.config.output_dir / 'plot.png')
            if d_real < 1e-4:
                log.info(f'EARLY STOP.')
                break

        elapsed = time.time() - start
        log.info (f'Training time: {hms_string(elapsed)}')

def build_generator(config: Config) -> models.Model:
    '''
    A generator network
    '''
    seed_size = config.seed_size
    channels = config.image_channels
    kernel_size = config.kernel_size
    alpha = config.generator_alpha
    size = config.image_size

    normal = initializers.RandomNormal(mean=0.0, stddev=0.02)

    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*256, use_bias=False, input_dim=seed_size))
    model.add(layers.Reshape((4, 4, 256)))
    assert model.output_shape == (None, 4, 4, 256)

    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(128, kernel_size=kernel_size, use_bias=False, 
        padding="same", kernel_initializer=normal))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=alpha))

    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(64, kernel_size=kernel_size, use_bias=False, 
        padding="same", kernel_initializer=normal))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=alpha))

    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(32, kernel_size=kernel_size, use_bias=False, 
        padding="same", kernel_initializer=normal))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=alpha))

    assert model.output_shape == (None, 32, 32, 32)
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(16, kernel_size=kernel_size, use_bias=False, 
        padding="same", kernel_initializer=normal))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=alpha))

    assert model.output_shape == (None, 64, 64, 16)
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(4, kernel_size=kernel_size, use_bias=False, 
        padding="same", kernel_initializer=normal))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=alpha))

    assert model.output_shape == (None, 128, 128, 4)
    model.add(layers.Conv2D(channels, kernel_size=kernel_size, use_bias=False, 
        padding="same", kernel_initializer=normal))

    model.add(layers.Activation('tanh'))
    return model

def build_discriminator(config: Config) -> models.Model:
    '''
    A discriminator network
    '''
    kernel_size = config.kernel_size
    alpha = config.discriminator_alpha
    momentum = config.momentum
    dropout = config.dropout
    image_shape = (config.image_size, config.image_size, config.image_channels)

    normal = initializers.RandomNormal(mean=0.0, stddev=0.02)

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, kernel_size=kernel_size, strides=2, 
        padding='same', kernel_initializer=normal, 
        input_shape=image_shape))
    model.add(layers.BatchNormalization(momentum=momentum, epsilon=1e-5))
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv2D(64, kernel_size=kernel_size, strides=2, 
        padding="same", kernel_initializer=normal))
    model.add(layers.BatchNormalization(momentum=momentum, epsilon=1e-5))
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv2D(128, kernel_size=kernel_size, strides=2, 
        padding="same", kernel_initializer=normal))
    model.add(layers.BatchNormalization(momentum=momentum, epsilon=1e-5))
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv2D(256, kernel_size=kernel_size, strides=2, 
        padding="same", kernel_initializer=normal))
    model.add(layers.BatchNormalization(momentum=momentum, epsilon=1e-5))
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Dropout(dropout))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model
