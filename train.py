import os

from keras.utils.vis_utils import plot_model
from tensorflow import keras

from clayers import FadeInCallBack, SamplingCallBack
from config import *
from dataset import ImageLoader
from modules import StyleGAN

# suppress tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# config check up
assert len(BSIZE) == len(FILTERS) and len(FILTERS) == len(EPOCHS), \
    f"BSIZE, FILTERS and EPOCHS must have the same size ({len(BSIZE)}, {len(FILTERS)}, {len(EPOCHS)})."


print(f'Latent vector dimension: {LDIM}')

image_loader = ImageLoader(INPUT_IMAGE_FOLDER, 'grayscale')

adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=EPS)

sgan = StyleGAN()
sgan.compile(d_optimizer=adam, g_optimizer=adam)

plot_model(sgan.G, to_file=os.path.join(MODEL_DIR, f'{SRES}x{SRES}_g_base.png'))
plot_model(sgan.D, to_file=os.path.join(MODEL_DIR, f'{SRES}x{SRES}_d_base.png'))

# callbacks
sampling_cbk = SamplingCallBack()

fade_in_cbk = FadeInCallBack()

# train initial models
training_images = image_loader.load(BSIZE[0], (SRES, SRES))
iters = len(training_images)
print(f"resolution: {SRES}x{SRES}, filters: {FILTERS[0]}")
sampling_cbk.set_prefix(f'{SRES}x{SRES}_base')
sgan.fit(training_images, steps_per_epoch=iters, epochs=EPOCHS[0], callbacks=[sampling_cbk])
sgan.save_weights(os.path.join(CKPTS_DIR, f'stylegan_{sampling_cbk.prefix}.ckpt'))

# grow and train models
for depth in range(1, len(BSIZE)):

    # grow model
    sgan.grow()                                            

    bs = BSIZE[depth]                             # batch size
    ep = EPOCHS[depth]                                 # epochs
    ch = FILTERS[depth]                                # filters
    rs = SRES * (2 ** depth)                               # resolution
    training_images = image_loader.load(bs, (rs, rs))  # load images
    iters = len(training_images)                       # iterations

    print(f'-- resolution: {rs}x{rs}, filters: {ch} --')

    # save model plots
    plot_model(sgan.G, to_file=os.path.join(MODEL_DIR, f'{rs}x{rs}_g_fadein.png'))
    plot_model(sgan.D, to_file=os.path.join(MODEL_DIR, f'{rs}x{rs}_d_fadein.png'))

    # fade in training
    sampling_cbk.set_prefix(f'{rs}x{rs}_fadein')
    fade_in_cbk.set_iters(ep, iters)
    sgan.compile(adam, adam)
    # additional callback to compute alpha
    sgan.fit(training_images, steps_per_epoch=iters, epochs=ep, callbacks=[sampling_cbk, fade_in_cbk])
    sgan.save_weights(os.path.join(CKPTS_DIR, f'stylegan_{sampling_cbk.prefix}.ckpt'))

    # transition from fade in models to complete high resolution models
    sgan.stabilize()

    # save model plots
    plot_model(sgan.G, to_file=os.path.join(MODEL_DIR, f'{rs}x{rs}_g_stabilize.png'))
    plot_model(sgan.D, to_file=os.path.join(MODEL_DIR, f'{rs}x{rs}_d_stabilize.png'))

    # stabilize training
    sampling_cbk.set_prefix(f'{rs}x{rs}_stabilize')
    sgan.compile(adam, adam)
    sgan.fit(training_images, steps_per_epoch=iters, epochs=ep, callbacks=[sampling_cbk])
    sgan.save_weights(os.path.join(CKPTS_DIR, f'stylegan_{sampling_cbk.prefix}.ckpt'))
