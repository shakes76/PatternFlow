import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

import config as cfg
from modules import StyleGAN


def load_model(ckpts_dir, sres, tres):
    depth = int(np.log2(tres/sres))
    # create model, load initial weights
    model = StyleGAN()
    model.load_weights(os.path.join(ckpts_dir, f'stylegan_{sres}x{sres}_base.ckpt'))

    # grow model, load weights
    for _ in range(depth):
        model.grow()
        model.stabilize()
    model.load_weights(os.path.join(ckpts_dir, f'stylegan_{tres}x{tres}_stabilize.ckpt'))
    print('Model loaded.')
    return model


def gen_inputs(model, latent_vec_dim, sres, tres, n=1, w=None):
    depth = int(np.log2(tres/sres))
    if w is None:
        z = tf.random.normal((n, latent_vec_dim))
        ws = model.FC(z)
    else:
        ws = w
    const = tf.ones([n, sres, sres, cfg.FILTERS[0]])
    inputs = [const]
    for i in range(depth+1):
        w = ws[:, i]
        B = tf.random.normal((n, sres*(2**i), sres*(2**i), 1))
        inputs += [w, B]
    return inputs


def plot_save(images, cols=None, plot=True, size=(256, 256), mode='L', save_path=None):
    n = len(images)
    w, h = size
    if cols == None:
        cols = int(np.sqrt(n))

    add = n % cols > 0
    n_w, n_h = cols, n // cols + (1 if add else 0)
    combined_image = Image.new(mode, (w * n_w, h * n_h))
    for i in range(n):
        image = tf.keras.preprocessing.image.array_to_img(images[i])
        image = image.resize(size)
        combined_image.paste(image, (i % n_w * w, i // n_w * h))

    if plot:
        # my monitor dpi
        dpi = 168.
        plt.figure(figsize=(n_w * w / dpi, n_h * h / dpi), dpi=dpi)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(combined_image, cmap='gray')

    if save_path is not None:
        combined_image.save(save_path)
        print(f'\n{n} images saved in {save_path}')


output_res = (260, 228)    # output resolution
sres = cfg.SRES
tres = cfg.TRES
ldim = cfg.LDIM

ckpts_dre = r'D:\AKOA\ckpts'                # folder of checkpoints

model = load_model(ckpts_dre, sres, tres)

# plot random images
n = 25
inputs = gen_inputs(model, ldim, sres, tres, n=n)
images = model.G(inputs)
plot_save(images, cols=5, size=output_res, save_path=r'D:\AKOA\images\generated.png')


# interpolation
w1 = model.FC(tf.random.normal((1, ldim)))
inputs = gen_inputs(model, ldim, sres, tres, w=w1)
images = model.G(inputs)
plot_save(images, size=output_res)


w2 = model.FC(tf.random.normal((1, ldim)))
inputs = gen_inputs(model, ldim, sres, tres, w=w2)
images = model.G(inputs)
plot_save(images, size=output_res)


l = []
steps = 25
for i in range(steps):
    alpha = (i + 1.) / steps
    l.append((1 - alpha) * w1 + alpha * w2)
w4 = tf.concat(l, axis=0)
inputs = gen_inputs(model, ldim, sres, tres, w=w4)
images = model.G(inputs)
plot_save(images, cols=5, size=output_res)
