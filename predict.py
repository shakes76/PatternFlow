import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.utils import array_to_img

from config import *
from modules import StyleGAN

# create model, load initial weights
sgan = StyleGAN(latent_dim=LATENT_VECTOR_DIM, filters=FILTERS, channels=CHANNELS)
sgan.load_weights(os.path.join(OUTPUT_CKPTS_FOLDER, f'stylegan_{SRES}x{SRES}_base.ckpt'))

depth = int(np.log2(TRES/SRES))

# grow model, load weights
for n_depth in range(depth):
    sgan.grow()
    sgan.stabilize()
sgan.load_weights(os.path.join(OUTPUT_CKPTS_FOLDER, f'stylegan_{TRES}x{TRES}_stabilize.ckpt'))

# number of samples
n = 25
# output resolution
res = 256

# build inputs
const = tf.ones([n, SRES, SRES, sgan.FILTERS[0]])
z = tf.random.normal((n, sgan.LDIM))
ws = sgan.FC(z)
inputs = [const]
for i in range(depth+1):
    w = ws[:, i]
    B = tf.random.normal((n, SRES*(2**i), SRES*(2**i), 1))
    inputs += [w, B]

samples = sgan.G(inputs)

w = h = int(np.sqrt(n))
out_imgs = Image.new('L', (res * w, res * h))
for i in range(n):
    img = array_to_img(samples[i]).resize((res, res))
    out_imgs.paste(img, (i % w * res, i // h * res))
plt.figure(figsize=(30, 30))
plt.xticks([])
plt.yticks([])
plt.imshow(out_imgs, cmap='gray')

path = os.path.join(OUTPUT_IMAGE_FOLDER, 'generated.png')
out_imgs.save(path)
print(f'\n{n} images saved: {path}')
