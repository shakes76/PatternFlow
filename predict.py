import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

import config as cfg
from modules import StyleGAN


def load_model(ckpts, sres, tres):
    depth = int(np.log2(tres/sres))
    # create model, load initial weights
    model = StyleGAN()
    # grow model, load weights
    for _ in range(depth):
        model.grow()
        model.stabilize()
    model.load_weights(os.path.join(ckpts))
    print('Model loaded.')
    return model.FC, model.G


def gen_inputs(fc, latent_vec_dim, sres, tres, n=1, w=None):
    depth = int(np.log2(tres/sres))
    if w is None:
        z = tf.random.normal((n, latent_vec_dim))
        ws = fc(z)
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


# suppress optimizer warning when loading checkpoints.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


sres = cfg.SRES           # starting resolution of the model
tres = cfg.TRES           # target resolution of the model
ldim = cfg.LDIM           # latent vector dimention of the model
output_res = (256, 256)   # output resolution of generated images
ckpt = r'path of ckpts'   # path of checkpoint files


# build model
FC, Synthesis = load_model(ckpt, sres, tres)


# plot 100 random images
n = 100
inputs = gen_inputs(FC, ldim, sres, tres, n=n)
images = Synthesis(inputs)
plot_save(images, size=output_res, save_path=r'D:\generated.png')


# bilinear interpolation
w1 = FC(tf.random.normal((1, ldim)))
w2 = FC(tf.random.normal((1, ldim)))
w3 = FC(tf.random.normal((1, ldim)))
w4 = FC(tf.random.normal((1, ldim)))
steps = 10
w12 = []
for i in range(steps):
    alpha = (i + 1.) / steps
    w12.append((1 - alpha) * w1 + alpha * w2)

w34 = []
for i in range(steps):
    alpha = (i + 1.) / steps
    w34.append((1 - alpha) * w3 + alpha * w4)

w1234 = []
for i in range(steps):
    for j in range(steps):
        alpha = (j + 1.) / steps
        w1234.append((1 - alpha) * w12[i] + alpha * w34[i])
w1234 = tf.concat(w1234, axis=0)
inputs = gen_inputs(FC, ldim, sres, tres, w=w1234)
images = Synthesis(inputs)
plot_save(images, cols=10, size=output_res, save_path=r'D:\bilinear_interpolation.png')
