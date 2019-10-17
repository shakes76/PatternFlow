#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import tensorflow as tf 
import time
import numpy as np


# Need tensorflow version >=2.0
print(tf.__version__)


# Load image in 
tf_img = tf.keras.preprocessing.image.load_img('dahlia.png')
cols, rows = tf_img.size

# Initialise S 
S = tf.Variable(np.array(tf_img), dtype=tf.float32)
S = S.assign(S/256)

size_2D = (rows, cols)
fx = tf.constant([1, -1], dtype=tf.int32)

fx = tf.constant([[1, -1]], dtype=tf.float32)
fy = tf.constant([[1], [-1]], dtype=tf.float32)


def psf2otf(psf, outSize):
    psfSize = tf.constant(psf.shape)

    new_psf = tf.Variable(tf.zeros(outSize, dtype=tf.float32))
    
    new_psf = new_psf[:psfSize[0],:psfSize[1]].assign(psf)
    psf = new_psf
    
    shift = -(psfSize / 2)
    print(shift.shape[0])
    print(shift)
    for i in range(shift.shape[0]):
        psf = tf.roll(psf, int(shift[i]), axis=i)
    
    otf = np.fft.fftn(psf)
    return otf


otfFx = psf2otf(fx, size_2D)
otfFy = psf2otf(fy, size_2D)


otfFx = tf.complex(otfFx.real, otfFx.imag)
otfFy = tf.complex(otfFy.real, otfFy.imag)


_lambda = 2e-2
kappa = 2.0

S_complex = tf.Variable(tf.complex(S, tf.constant(tf.zeros((rows, cols, 3)))))
S_complex = tf.cast(S_complex, dtype=tf.complex128)
S_complex = tf.Variable(S_complex)

FI = tf.Variable(tf.complex(tf.zeros((rows, cols, 3)), tf.zeros((rows, cols, 3))))

FI = tf.cast(FI, dtype=tf.complex128)
FI = tf.Variable(FI)
for i in range(S_complex.shape[2]):
    FI = FI[:,:,i].assign(tf.cast(tf.signal.fft2d(S_complex[:,:,i]), dtype=tf.complex128))


MTF = tf.math.square(tf.math.abs(otfFx)) + tf.math.square(tf.math.abs(otfFy))
MTF = tf.stack([MTF, MTF, MTF], axis=2)


h = tf.Variable(tf.zeros((rows, cols, 3), dtype=tf.float64))
v = tf.Variable(tf.zeros((rows, cols, 3), dtype=tf.float64))
dxhp = tf.Variable(tf.zeros((rows, cols, 3), dtype=tf.float64))
dyvp = tf.Variable(tf.zeros((rows, cols, 3), dtype=tf.float64))
FS = tf.Variable(tf.complex(tf.zeros((rows, cols, 3), dtype=tf.float64), 
                            tf.zeros((rows, cols, 3), dtype=tf.float64) ))
FS = tf.cast(FS, dtype=tf.complex128)

beta_max = 1e5;
beta = 2 * _lambda
iteration = 0


M = cols
N = rows 


while beta < beta_max:
    h = tf.Variable(h)
    v = tf.Variable(v)

    h = h[:,0:M-1,:].assign(tf.cast(tf.math.real(S_complex[:,1:]-S_complex[:,:-1]), dtype=tf.float64))
    h = h[:,M-1:M,:].assign(tf.cast(tf.math.real(S_complex[:,0:1,:] - S_complex[:,M-1:M,:]), dtype=tf.float64))

    v = v[0:N-1,:,:].assign(tf.cast(tf.math.real(S_complex[1:]-S_complex[:-1]), dtype=tf.float64))
    v = v[N-1:N,:,:].assign(tf.cast(tf.math.real(S_complex[0:1,:,:] - S_complex[N-1:N,:,:]), dtype=tf.float64))
    
    t = tf.reduce_sum(tf.math.square(h) + tf.math.square(v), axis=2) < _lambda / beta
    t = tf.stack([t, t, t], axis=2)

    idx = tf.where(t)
    h = tf.tensor_scatter_nd_update(h, idx, tf.zeros(idx.shape[0], dtype=tf.float64))
    v = tf.tensor_scatter_nd_update(v, idx, tf.zeros(idx.shape[0], dtype=tf.float64))

    dxhp = dxhp[:,0:1,:].assign(h[:,M-1:M,:] - h[:,0:1,:])
    dxhp = dxhp[:,1:M,:].assign(-(h[:,1:]-h[:,:-1]))
    dyvp = dyvp[0:1,:,:].assign(v[N-1:N,:,:] - v[0:1,:,:])
    dyvp = dyvp[1:N,:,:].assign(-(v[1:]-v[:-1]))
    normin = dxhp + dyvp
    
    normin = tf.complex(normin, tf.constant(tf.zeros(normin.shape, dtype=tf.float64)))
    for i in range(S.shape[2]):
        FS = FS[:,:,i].assign(tf.signal.fft2d(normin[:,:,i]))
    
    denorm = tf.cast(1 + beta * MTF, dtype=tf.complex128)
    FS = FS[:,:,:].assign((FI + beta * FS) / denorm)
    
    for i in range(S.shape[2]):
        S_complex = tf.Variable(S_complex) 
        S_complex = S_complex[:,:,i].assign(tf.signal.ifft2d(FS[:,:,i]))

    beta *= kappa
    iteration += 1

# Rescale image
S_complex = S_complex * 256

print("Iterations: %d" % (iteration))


im = Image.fromarray(tf.math.real(S_complex).numpy().astype(np.uint8))
im.save("out2.png")





