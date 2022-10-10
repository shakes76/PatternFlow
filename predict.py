import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

from config import *
from modules import StyleGAN

sgan = StyleGAN(latent_dim=LATENT_VECTOR_DIM, filters=FILTERS, channels=CHANNELS)

sgan.load_weights(OUTPUT_CKPTS_FOLDER + f'/pgan_{sgan.SRES}x{sgan.SRES}_init.ckpt')


for n_depth in range(1, 7):
    res = sgan.SRES*(2**n_depth)
    print(f'{res}x{res}')
    sgan.current_depth = n_depth
    prefix = f'{res}x{res}_fadein'
    sgan.grow()
    sgan.load_weights(OUTPUT_CKPTS_FOLDER + f'/pgan_{prefix}.ckpt')

    prefix = f'{res}x{res}_transition'
    sgan.stabilize_generator()
    sgan.stabilize_discriminator()
    sgan.load_weights(OUTPUT_CKPTS_FOLDER + f'/pgan_{prefix}.ckpt')
sgan.load_weights(OUTPUT_CKPTS_FOLDER + f'/pgan_{prefix}.ckpt')

n = 100
res = 256
z = tf.random.normal((n, LATENT_VECTOR_DIM))
w = sgan.FC(z)
const = tf.ones([n, sgan.SRES, sgan.SRES, sgan.FILTERS[0]])
samples = sgan.G([const, w])

wh = int(np.sqrt(n))
out_imgs = Image.new('L', (res * wh, res * wh))
for i in range(n):
    img = tf.keras.preprocessing.image.array_to_img(samples[i])
    img = img.resize((res, res))
    out_imgs.paste(img, (i % wh * res, i // wh * res))
plt.figure(figsize=(30, 30))
plt.xticks([])
plt.yticks([])
plt.imshow(out_imgs, cmap='Greys_r')
