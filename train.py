import os
import time

from keras.utils.vis_utils import plot_model
from numpy import array, log2
from tensorflow import keras

from clayers import FadeInCallBack, SamplingCallBack
from config import *
from dataset import ImageLoader
from modules import StyleGAN


def main():
    
    print('Starting training...')
    
    print(f'Training image folder: {TRAINING_IMAGE_DIR}')
    
    # suppress tensorflow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # config check up
    assert len(BSIZE) == len(FILTERS) and len(FILTERS) == len(EPOCHS) and len(EPOCHS) == int(log2(TRES) - log2(SRES) + 1), \
        f'BSIZE, FILTERS and EPOCHS must have the same size ({len(BSIZE)}, {len(FILTERS)}, {len(EPOCHS[0])}), ' \
        'and their size must equal log2(TRES)+1.'

    # output folder must not exist
    assert not os.path.exists(OUT_ROOT), \
        f'Folder \'{OUT_ROOT}\' already exists. Specify another folder for \'OUT_ROOT\' in config.py'

    print(f'Creating output folders: {OUT_ROOT}')
    # create folders
    os.mkdir(OUT_ROOT)
    time.sleep(2)
    assert os.path.exists(OUT_ROOT), \
        f'Output folder {OUT_ROOT} was not created successfully.'
    print(f'Root folder {OUT_ROOT} created.')
    subfolders = {
        'ckpts': os.path.join(OUT_ROOT, 'ckpts'),     # check points folder
        'images': os.path.join(OUT_ROOT, 'images'),   # output image folder
        'log': os.path.join(OUT_ROOT, 'log'),         # loss history csv folder
        'models': os.path.join(OUT_ROOT, 'models'),   # output model plot folder
    }
    for _, subfolder in subfolders.items():
        os.mkdir(subfolder)
        time.sleep(2)
        assert os.path.exists(subfolder), f'sub-folder {subfolder} was not created.'
        print(f'Sub-folder {subfolder} created.')

    print(f'Latent vector dimension: {LDIM}')

    image_loader = ImageLoader(TRAINING_IMAGE_DIR, 'grayscale')

    adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1.e-8)

    model = StyleGAN()
    model.compile(d_optimizer=adam, g_optimizer=adam)

    plot_model(model.FC, to_file=os.path.join(subfolders['models'], 'fc.png'), rankdir='LR')
    plot_model(model.G, to_file=os.path.join(subfolders['models'], f'{SRES}x{SRES}_g_base.png'), rankdir='LR')
    plot_model(model.D, to_file=os.path.join(subfolders['models'], f'{SRES}x{SRES}_d_base.png'), rankdir='LR')

    # callbacks
    sampling_cbk = SamplingCallBack(
        output_img_res=(256, 240),  # some provided images are not square, resize to match.
        output_img_folder=subfolders['images'], 
        output_ckpts_folder=subfolders['ckpts'])
    fade_in_cbk = FadeInCallBack()

    # train initial models
    training_images = image_loader.load(BSIZE[0], (SRES, SRES))
    iters = len(training_images)
    print(f"-- resolution: {SRES}x{SRES}, filters: {FILTERS[0]} --")
    sampling_cbk.set_prefix(f'{SRES}x{SRES}_base')

    loss_history = []

    hist_init = model.fit(training_images, steps_per_epoch=iters, epochs=EPOCHS[0], callbacks=[sampling_cbk])
    loss_history.append(hist_init)
    model.save_weights(os.path.join(subfolders['ckpts'], f'stylegan_{sampling_cbk.prefix}.ckpt'))

    # grow and train models
    for depth in range(1, len(BSIZE)):

        # grow model
        model.grow()

        bs = BSIZE[depth]                                  # batch size
        ep = EPOCHS[depth][0]                              # epochs
        ch = FILTERS[depth]                                # filters
        rs = SRES * (2 ** depth)                           # resolution
        training_images = image_loader.load(bs, (rs, rs))  # load images
        iters = len(training_images)                       # iterations

        print(f'-- resolution: {rs}x{rs}, filters: {ch} --')

        # save model plot
        plot_model(model.G, to_file=os.path.join(subfolders['models'], f'{rs}x{rs}_g_fadein.png'), rankdir='LR')
        plot_model(model.D, to_file=os.path.join(subfolders['models'], f'{rs}x{rs}_d_fadein.png'), rankdir='LR')

        # fade in training
        sampling_cbk.set_prefix(f'{rs}x{rs}_fadein')
        fade_in_cbk.set_iters(ep, iters)
        model.compile(adam, adam)
        # additional callback to compute alpha
        hist_fade = model.fit(training_images, steps_per_epoch=iters, epochs=ep, callbacks=[sampling_cbk, fade_in_cbk])
        loss_history.append(hist_fade)
        model.save_weights(os.path.join(subfolders['ckpts'], f'stylegan_{sampling_cbk.prefix}.ckpt'))

        model.stabilize()
        
        # whether to stabilize-train, this should be specific to training image
        # can lead to over train (model collapse)
        if STAB:
            # transition from fade in models to complete high resolution models
            
            ep = EPOCHS[depth][1]

            # save model plots
            plot_model(model.G, to_file=os.path.join(subfolders['models'], f'{rs}x{rs}_g_stabilize.png'), rankdir='LR')
            plot_model(model.D, to_file=os.path.join(subfolders['models'], f'{rs}x{rs}_d_stabilize.png'), rankdir='LR')

            # stabilize training
            sampling_cbk.set_prefix(f'{rs}x{rs}_stabilize')
            model.compile(adam, adam)
            hist_stab = model.fit(training_images, steps_per_epoch=iters, epochs=ep, callbacks=[sampling_cbk])
            loss_history.append(hist_stab)
            model.save_weights(os.path.join(subfolders['ckpts'], f'stylegan_{sampling_cbk.prefix}.ckpt'))

    print('\nTraining completed.')

    # save loss logs
    D_loss = []
    G_loss = []
    for hist in loss_history:
        D_loss += hist.history['d_loss']
        G_loss += hist.history['g_loss']

    D_log_csv = os.path.join(subfolders['log'], 'dloss.csv')
    G_log_csv = os.path.join(subfolders['log'], 'gloss.csv')
    array(D_loss).tofile(D_log_csv, sep=',')
    array(G_loss).tofile(G_log_csv, sep=',')

    print(f'{D_log_csv} saved.')
    print(f'{G_log_csv} saved.')


if __name__ == "__main__":
    main()
