from modules import StyleGAN, WeightedSum
from dataset import ImageLoader
from tensorflow import keras
from PIL import Image
from keras import backend
import tensorflow as tf
import numpy as np
import os
from keras.utils.vis_utils import plot_model

from config import *

# suppress tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# OOM
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# config validity check
assert len(BATCH_SIZE) == len(FILTERS) and len(FILTERS) == len(EPOCHS), \
    f"BATCH_SIZE, FILTERS and EPOCHS must have the same size ({len(BATCH_SIZE)}, {len(FILTERS)}, {len(EPOCHS)})."

class SGCallBack(tf.keras.callbacks.Callback):

    def __init__(
        self, 
        latent_dim=100,
        current_depth=0,
        output_num_img=16,
        output_img_dim=256,
        output_img_folder='',
        output_ckpts_folder='',
        is_output_rgb=True,
        prefix=''
    ):
        self.output_num_img = output_num_img
        self.latent_dim = latent_dim
        self.current_depth = current_depth
        self.z = tf.random.normal((self.output_num_img, self.latent_dim), seed=42)
        self.steps_per_epoch = 0
        self.epochs = 0
        self.steps = self.steps_per_epoch * self.epochs
        self.n_epoch = 0
        self.prefix = prefix

        self.output_img_dim = output_img_dim
        self.output_img_folder = output_img_folder
        self.output_img_mode = 'RGB' if is_output_rgb else 'L'

        self.output_ckpts_folder = output_ckpts_folder

    def set_prefix(self, prefix=''):
        self.prefix = prefix
    
    def set_currentDepth(self, depth=0):
        self.current_depth = depth

    def set_steps(self, steps_per_epoch, epochs):
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.steps = self.steps_per_epoch * self.epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.n_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        sgan = self.model
        w = sgan.FC(self.z)
        ws = [w[:,i] for i in range(self.current_depth+1)]
        const = tf.ones([self.output_num_img, sgan.SRES, sgan.SRES, sgan.FILTERS[0]])
        samples = sgan.G([const]+ws)

        wh = int(np.sqrt(self.output_num_img))
        out_imgs = Image.new(self.output_img_mode, (self.output_img_dim * wh, self.output_img_dim * wh))
        for i in range(self.output_num_img):
            img = tf.keras.preprocessing.image.array_to_img(samples[i])
            img = img.resize((self.output_img_dim, self.output_img_dim))
            out_imgs.paste(img, (i % wh * self.output_img_dim, i // wh * self.output_img_dim))
        path = os.path.join(self.output_img_folder, f'{self.prefix}_{epoch+1:02d}.png')
        out_imgs.save(path)
        print(f'\n{self.output_num_img} progress images saved: {path}')

    def on_batch_begin(self, batch, logs=None):
        # Update alpha in WeightedSum layers
        alpha = ((self.n_epoch * self.steps_per_epoch) + batch) / float(self.steps - 1)
        for layer in self.model.G.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)
        for layer in self.model.D.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)

print(f'Latent vector dimension: {LATENT_VECTOR_DIM}')

image_loader = ImageLoader(INPUT_IMAGE_FOLDER, 'grayscale')

adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

# Instantiate the PGAN(PG-GAN) model.
sgan = StyleGAN(latent_dim=LATENT_VECTOR_DIM, filters=FILTERS, channels=CHANNELS)
# Compile models
sgan.compile(d_optimizer=adam, g_optimizer=adam)

plot_model(sgan.G, to_file=os.path.join(OUTPUT_MODEL_FOLDER, f'{sgan.SRES}x{sgan.SRES}_g_base.png'))
plot_model(sgan.D, to_file=os.path.join(OUTPUT_MODEL_FOLDER, f'{sgan.SRES}x{sgan.SRES}_d_base.png'))

cbk = SGCallBack(
    latent_dim=LATENT_VECTOR_DIM,
    output_num_img=N_SAMPLES,
    output_img_folder=OUTPUT_IMAGE_FOLDER,
    output_ckpts_folder=OUTPUT_CKPTS_FOLDER,
    is_output_rgb=False)

training_images = image_loader.load(BATCH_SIZE[0], (sgan.SRES, sgan.SRES))
st = len(training_images)

print(f"resolution: {sgan.SRES}x{sgan.SRES}, filters: {FILTERS[0]}")

cbk.set_prefix(f'{sgan.SRES}x{sgan.SRES}_base')
cbk.set_currentDepth(0)
cbk.set_steps(steps_per_epoch=st, epochs=EPOCHS[0])
sgan.fit(training_images, steps_per_epoch=st, epochs=EPOCHS[0], callbacks=[cbk])
sgan.save_weights(os.path.join(OUTPUT_CKPTS_FOLDER, f'stylegan_{cbk.prefix}.ckpt'))

for depth in range(1, len(BATCH_SIZE)):
    
    sgan.current_depth = depth                             # set current depth
    sgan.grow()                                            # grow model
    
    bs = BATCH_SIZE[depth]                                 # batch size
    ep = EPOCHS[depth]                                     # epochs 
    ch = FILTERS[depth]                                    # filters
    res = sgan.SRES*(2**depth)                             # resolution
    training_images = image_loader.load(bs, (res, res))    # load images
    st = len(training_images)                              # steps
    
    print(f"res: {res}x{res}, filters: {ch}")

    cbk.set_steps(steps_per_epoch=st, epochs=ep)
    cbk.set_currentDepth(depth)
    cbk.set_prefix(f'{res}x{res}_fadein')
    plot_model(sgan.G, to_file=os.path.join(OUTPUT_MODEL_FOLDER, f'{res}x{res}_g_fadein.png'))
    plot_model(sgan.D, to_file=os.path.join(OUTPUT_MODEL_FOLDER, f'{res}x{res}_d_fadein.png'))
    sgan.compile(adam, adam)
    sgan.fit(training_images, steps_per_epoch=st, epochs=ep, callbacks=[cbk])
    sgan.save_weights(os.path.join(OUTPUT_CKPTS_FOLDER, f'stylegan_{cbk.prefix}.ckpt'))

    sgan.transition()

    cbk.set_prefix(f'{res}x{res}_trans')
    plot_model(sgan.G, to_file=os.path.join(OUTPUT_MODEL_FOLDER, f'{res}x{res}_g_trans.png'))
    plot_model(sgan.D, to_file=os.path.join(OUTPUT_MODEL_FOLDER, f'{res}x{res}_d_trans.png'))
    sgan.compile(adam, adam)
    sgan.fit(training_images, steps_per_epoch=st, epochs=ep, callbacks=[cbk])
    sgan.save_weights(os.path.join(OUTPUT_CKPTS_FOLDER, f'stylegan_{cbk.prefix}.ckpt'))
