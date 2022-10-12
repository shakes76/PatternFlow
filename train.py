import os

from keras.utils.vis_utils import plot_model
from numpy import array, concatenate
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

model = StyleGAN()
model.compile(d_optimizer=adam, g_optimizer=adam)

plot_model(model.G, to_file=os.path.join(MODEL_DIR, f'{SRES}x{SRES}_g_base.png'))
plot_model(model.D, to_file=os.path.join(MODEL_DIR, f'{SRES}x{SRES}_d_base.png'))

# callbacks
sampling_cbk = SamplingCallBack()
fade_in_cbk = FadeInCallBack()

# train initial models
training_images = image_loader.load(BSIZE[0], (SRES, SRES))
iters = len(training_images)
print(f"-- resolution: {SRES}x{SRES}, filters: {FILTERS[0]} --")
sampling_cbk.set_prefix(f'{SRES}x{SRES}_base')

loss_history = []

hist_init = model.fit(training_images, steps_per_epoch=iters, epochs=EPOCHS[0], callbacks=[sampling_cbk])
loss_history.append(hist_init)
model.save_weights(os.path.join(CKPTS_DIR, f'stylegan_{sampling_cbk.prefix}.ckpt'))

# grow and train models
for depth in range(1, len(BSIZE)):

    # grow model
    model.grow()                                            

    bs = BSIZE[depth]                                  # batch size
    ep = EPOCHS[depth]                                 # epochs
    ch = FILTERS[depth]                                # filters
    rs = SRES * (2 ** depth)                           # resolution
    training_images = image_loader.load(bs, (rs, rs))  # load images
    iters = len(training_images)                       # iterations

    print(f'-- resolution: {rs}x{rs}, filters: {ch} --')

    # save model plots
    plot_model(model.G, to_file=os.path.join(MODEL_DIR, f'{rs}x{rs}_g_fadein.png'))
    plot_model(model.D, to_file=os.path.join(MODEL_DIR, f'{rs}x{rs}_d_fadein.png'))

    # fade in training
    sampling_cbk.set_prefix(f'{rs}x{rs}_fadein')
    fade_in_cbk.set_iters(ep, iters)
    model.compile(adam, adam)
    # additional callback to compute alpha
    hist_fade = model.fit(training_images, steps_per_epoch=iters, epochs=int((ep*.7)+1), callbacks=[sampling_cbk, fade_in_cbk])
    loss_history.append(hist_fade)
    model.save_weights(os.path.join(CKPTS_DIR, f'stylegan_{sampling_cbk.prefix}.ckpt'))

    # transition from fade in models to complete high resolution models
    model.stabilize()

    # save model plots
    plot_model(model.G, to_file=os.path.join(MODEL_DIR, f'{rs}x{rs}_g_stabilize.png'))
    plot_model(model.D, to_file=os.path.join(MODEL_DIR, f'{rs}x{rs}_d_stabilize.png'))

    # stabilize training
    sampling_cbk.set_prefix(f'{rs}x{rs}_stabilize')
    model.compile(adam, adam)
    hist_stab = model.fit(training_images, steps_per_epoch=iters, epochs=int((ep*.3)+1), callbacks=[sampling_cbk])
    loss_history.append(hist_stab)
    model.save_weights(os.path.join(CKPTS_DIR, f'stylegan_{sampling_cbk.prefix}.ckpt'))

print('training completed.')

# save loss logs
G_loss = []
D_loss = []
for hist in loss_history:
    G_loss += hist.history['d_loss'] 
    D_loss += hist.history['g_loss']

array(G_loss).tofile(os.path.join(LOG_DIR, 'dloss.csv'), sep=',')
array(D_loss).tofile(os.path.join(LOG_DIR, 'gloss.csv'), sep=',')
