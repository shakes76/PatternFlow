import os

import numpy as np
import tensorflow as tf
from keras import backend
from keras.utils.vis_utils import plot_model
from PIL import Image
from tensorflow import keras

from config import *
from dataset import ImageLoader
from modules import StyleGAN, WeightedSum

# suppress tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# OOM
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# config check up
assert len(BATCH_SIZE) == len(FILTERS) and len(FILTERS) == len(EPOCHS), \
    f"BATCH_SIZE, FILTERS and EPOCHS must have the same size ({len(BATCH_SIZE)}, {len(FILTERS)}, {len(EPOCHS)})."


class SamplingCallBack(tf.keras.callbacks.Callback):
    """
    callback for saving progressive images after epoch
    """

    def __init__(
        self,
        output_num_img=16,
        output_img_res=256,
        output_img_folder='',
        output_ckpts_folder='',
        is_rgb=True,
    ):
        self.output_num_img = output_num_img              # number of output images
        self.output_img_dim = output_img_res              # output image resolution/size
        self.output_img_mode = 'RGB' if is_rgb else 'L'   # output image mode. color or black white?

        self.output_img_folder = output_img_folder        # output image folder
        self.output_ckpts_folder = output_ckpts_folder    # checkpoints foler

    def set_prefix(self, prefix=''):
        self.prefix = prefix

    def set_current_depth(self, current_depth):
        self.current_depth = current_depth

    def on_epoch_end(self, epoch, logs=None):
        sgan = self.model

        # build inputs for G
        const = tf.ones([self.output_num_img, sgan.SRES, sgan.SRES, sgan.FILTERS[0]])
        z = tf.random.normal((self.output_num_img, sgan.LDIM), seed=3710)
        ws = sgan.FC(z)
        inputs = [const]
        for i in range(sgan.current_depth+1):
            w = ws[:, i]
            B = tf.random.normal((self.output_num_img, sgan.SRES*(2**i), sgan.SRES*(2**i), 1))
            inputs += [w, B]

        # generate
        samples = sgan.G(inputs)

        # save
        wh = int(np.sqrt(self.output_num_img))
        out_imgs = Image.new(self.output_img_mode, (self.output_img_dim * wh, self.output_img_dim * wh))
        for i in range(self.output_num_img):
            img = tf.keras.preprocessing.image.array_to_img(samples[i])
            img = img.resize((self.output_img_dim, self.output_img_dim))
            out_imgs.paste(img, (i % wh * self.output_img_dim, i // wh * self.output_img_dim))
        path = os.path.join(self.output_img_folder, f'{self.prefix}_{epoch+1:02d}.png')
        out_imgs.save(path)
        print(f'\n{self.output_num_img} progress images saved: {path}')


class FadeInCallBack(tf.keras.callbacks.Callback):
    """
    callback for increasing alpha from 0 to one
    only used for generator fade in phase
    """

    def __init__(self):
        # total iters = epochs * steps_per_epoch
        self.iters_per_epoch = 0
        self.epochs = 0
        self.iters = 0
        self.current_epoch = 0

    def set_iters(self, epochs, iters_per_epoch):
        self.epochs = epochs
        self.steps_per_epoch = iters_per_epoch
        self.iters = epochs * iters_per_epoch

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_begin(self, current_iter, logs=None):
        # update alpha for fade-in layers
        # for each resolution:
        #     alpha = ((current epoch - 1) * steps per epoch + current iteration) / (total iterations)
        alpha = ((self.current_epoch * self.steps_per_epoch) + current_iter + 1) / float(self.iters)
        if (1 == 1):
            pass
        for layer in self.model.G.layers + self.model.D.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)


print(f'Latent vector dimension: {LATENT_VECTOR_DIM}')

image_loader = ImageLoader(INPUT_IMAGE_FOLDER, 'grayscale')

adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

sgan = StyleGAN(latent_dim=LATENT_VECTOR_DIM, filters=FILTERS, channels=CHANNELS, sres=SRES, tres=TRES)
sgan.compile(d_optimizer=adam, g_optimizer=adam)

plot_model(sgan.G, to_file=os.path.join(OUTPUT_MODEL_FOLDER, f'{sgan.SRES}x{sgan.SRES}_g_base.png'))
plot_model(sgan.D, to_file=os.path.join(OUTPUT_MODEL_FOLDER, f'{sgan.SRES}x{sgan.SRES}_d_base.png'))

# callbacks
sampling_cbk = SamplingCallBack(
    latent_dim=LATENT_VECTOR_DIM,
    output_num_img=N_SAMPLES,
    output_img_folder=OUTPUT_IMAGE_FOLDER,
    output_ckpts_folder=OUTPUT_CKPTS_FOLDER,
    is_rgb=False)
fade_in_cbk = FadeInCallBack()

training_images = image_loader.load(BATCH_SIZE[0], (sgan.SRES, sgan.SRES))
iters = len(training_images)

print(f"resolution: {sgan.SRES}x{sgan.SRES}, filters: {FILTERS[0]}")

sampling_cbk.set_prefix(f'{sgan.SRES}x{sgan.SRES}_base')
sgan.fit(training_images, steps_per_epoch=iters, epochs=EPOCHS[0], callbacks=[sampling_cbk])
sgan.save_weights(os.path.join(OUTPUT_CKPTS_FOLDER, f'stylegan_{sampling_cbk.prefix}.ckpt'))

for depth in range(1, len(BATCH_SIZE)):

    sgan.grow()                                            # grow model

    bs = BATCH_SIZE[depth]                                 # batch size
    ep = EPOCHS[depth]                                     # epochs
    ch = FILTERS[depth]                                    # filters
    res = sgan.SRES*(2**depth)                             # resolution
    training_images = image_loader.load(bs, (res, res))    # load images
    iters = len(training_images)                           # iterations

    print(f"resolution: {res}x{res}, filters: {ch}")

    # save model plots
    plot_model(sgan.G, to_file=os.path.join(OUTPUT_MODEL_FOLDER, f'{res}x{res}_g_fadein.png'))
    plot_model(sgan.D, to_file=os.path.join(OUTPUT_MODEL_FOLDER, f'{res}x{res}_d_fadein.png'))

    sampling_cbk.set_prefix(f'{res}x{res}_fadein')
    fade_in_cbk.set_iters(ep, iters)

    # fade in training
    sgan.compile(adam, adam)
    sgan.fit(training_images, steps_per_epoch=iters, epochs=ep, callbacks=[sampling_cbk, fade_in_cbk])
    sgan.save_weights(os.path.join(OUTPUT_CKPTS_FOLDER, f'stylegan_{sampling_cbk.prefix}.ckpt'))

    sgan.stabilize()

    sampling_cbk.set_prefix(f'{res}x{res}_stabilize')

    # save model plots
    plot_model(sgan.G, to_file=os.path.join(OUTPUT_MODEL_FOLDER, f'{res}x{res}_g_stabilize.png'))
    plot_model(sgan.D, to_file=os.path.join(OUTPUT_MODEL_FOLDER, f'{res}x{res}_d_stabilize.png'))

    # stabilize training
    sgan.compile(adam, adam)
    sgan.fit(training_images, steps_per_epoch=iters, epochs=ep, callbacks=[sampling_cbk])
    sgan.save_weights(os.path.join(OUTPUT_CKPTS_FOLDER, f'stylegan_{sampling_cbk.prefix}.ckpt'))
