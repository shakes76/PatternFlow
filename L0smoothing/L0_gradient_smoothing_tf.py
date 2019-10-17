# Chosen algorithm: L0 norm smoothing of images
# author: naziah siddique 

from PIL import Image
import tensorflow as tf 
import numpy as np
import argparse 

# Require tensorflow version >=2.0 to run
print(tf.__version__)

def psf2otf(psf, outSize):
    psfSize = tf.constant(psf.shape)

    new_psf = tf.Variable(tf.zeros(outSize, dtype=tf.float32))
    
    new_psf = new_psf[:psfSize[0],:psfSize[1]].assign(psf)
    psf = new_psf
    
    shift = -(psfSize / 2)
    for i in range(shift.shape[0]):
        psf = tf.roll(psf, int(shift[i]), axis=i)
    
    otf = np.fft.fftn(psf)
    return otf

def l0_calc(img_arr, _lambda=2e-2, kappa=2.0, beta_max=1e5):
    N = img_arr.shape[0]
    M = img_arr.shape[1]
    D = img_arr.shape[2]

    rows = N
    cols = M

    # Initialise S 
    S = tf.Variable(img_arr, dtype=tf.float32)
    S = S.assign(S/256)

    size_2D = (rows, cols)
    fx = tf.constant([1, -1], dtype=tf.int32)

    fx = tf.constant([[1, -1]], dtype=tf.float32)
    fy = tf.constant([[1], [-1]], dtype=tf.float32)


    otfFx = psf2otf(fx, size_2D)
    otfFy = psf2otf(fy, size_2D)


    otfFx = tf.complex(otfFx.real, otfFx.imag)
    otfFy = tf.complex(otfFy.real, otfFy.imag)

    S_complex = tf.Variable(tf.complex(S, tf.constant(tf.zeros((rows, cols, 3)))))
    S_complex = tf.cast(S_complex, dtype=tf.complex128)
    S_complex = tf.Variable(S_complex)

    FI = tf.Variable(tf.complex(tf.zeros((rows, cols, D)), tf.zeros((rows, cols, 3))))

    FI = tf.cast(FI, dtype=tf.complex128)
    FI = tf.Variable(FI)
    for i in range(S_complex.shape[2]):
        FI = FI[:,:,i].assign(tf.cast(tf.signal.fft2d(S_complex[:,:,i]), dtype=tf.complex128))


    MTF = tf.math.square(tf.math.abs(otfFx)) + tf.math.square(tf.math.abs(otfFy))
    MTF = tf.stack([MTF, MTF, MTF], axis=2)


    h = tf.Variable(tf.zeros((rows, cols, D), dtype=tf.float64))
    v = tf.Variable(tf.zeros((rows, cols, D), dtype=tf.float64))
    dxhp = tf.Variable(tf.zeros((rows, cols, D), dtype=tf.float64))
    dyvp = tf.Variable(tf.zeros((rows, cols, D), dtype=tf.float64))
    FS = tf.Variable(tf.complex(tf.zeros((rows, cols, D), dtype=tf.float64), 
                                tf.zeros((rows, cols, D), dtype=tf.float64) ))
    FS = tf.cast(FS, dtype=tf.complex128)

    beta = 2 * _lambda  # set initial beta value
    iterations = 0               # used to track number of iterations 

    while beta < beta_max:
        h = tf.Variable(h)
        v = tf.Variable(v)

        h = h[:,0:cols-1,:].assign(
            tf.cast(tf.math.real(S_complex[:,1:]-S_complex[:,:-1]), dtype=tf.float64))
        h = h[:,cols-1:cols,:].assign(
            tf.cast(tf.math.real(S_complex[:,0:1,:]-S_complex[:,cols-1:cols,:]), dtype=tf.float64))

        v = v[0:rows-1,:,:].assign(
            tf.cast(tf.math.real(S_complex[1:]-S_complex[:-1]), dtype=tf.float64))
        v = v[rows-1:rows,:,:].assign(
            tf.cast(tf.math.real(S_complex[0:1,:,:] - S_complex[rows-1:rows,:,:]), dtype=tf.float64))
        
        t = tf.reduce_sum(
            tf.math.square(h) + tf.math.square(v), axis=2) < _lambda / beta
        t = tf.stack([t, t, t], axis=2)

        idx = tf.where(t)
        h = tf.tensor_scatter_nd_update(h, idx, tf.zeros(idx.shape[0], dtype=tf.float64))
        v = tf.tensor_scatter_nd_update(v, idx, tf.zeros(idx.shape[0], dtype=tf.float64))

        dxhp = dxhp[:,0:1,:].assign(h[:,cols-1:cols,:] - h[:,0:1,:])
        dxhp = dxhp[:,1:cols,:].assign(-(h[:,1:]-h[:,:-1]))
        dyvp = dyvp[0:1,:,:].assign(v[rows-1:rows,:,:] - v[0:1,:,:])
        dyvp = dyvp[1:rows,:,:].assign(-(v[1:]-v[:-1]))
        normin = dxhp + dyvp
        
        normin = tf.complex(
            normin, tf.constant(tf.zeros(normin.shape, dtype=tf.float64)))

        for i in range(S.shape[2]):
            FS = FS[:,:,i].assign(tf.signal.fft2d(normin[:,:,i]))
        
        denorm = tf.cast(1 + beta * MTF, dtype=tf.complex128)
        FS = FS[:,:,:].assign((FI + beta * FS) / denorm)
        
        # compute inverse fourier transform 
        for i in range(S.shape[2]):
            S_complex = tf.Variable(S_complex) 
            S_complex = S_complex[:,:,i].assign(tf.signal.ifft2d(FS[:,:,i]))

        beta *= kappa           # increase beta value kappa times 
        iterations += 1

    # Rescale image
    S_complex = S_complex * 256

    print("Iterations made: %d" % (iterations))

    # convert real S values to array
    output_arr = tf.math.real(S_complex).numpy().astype(np.uint8)

    return output_arr


def main(imdir, outdir, _lambda, kappa, beta_max):

    # load image into array 
    tf_img = tf.keras.preprocessing.image.load_img(imdir)
    img_arr = np.array(tf_img)

    # pass image and calculate and output gradient smoothing 
    out_img = l0_calc(img_arr, _lambda, kappa, beta_max)
    
    # save image from output array 
    im = Image.fromarray(out_img)
    im.save(outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="L0 Gradient Smoothing")
    parser.add_argument("-d", "--inputimgdir", dest="imgdir",
                        help="Directory path for input image",
                        metavar="FILE", default='data/dahlia.png')
    parser.add_argument("-o", "--outdir", dest="outdir",
                        help="Directory path for output image",
                        metavar="FILE", default="out.png")
    parser.add_argument("-l", "--lamdaval", dest="lamdaval",
                        help="lambda parameter",
                        metavar="FLOAT", default=2e-2)
    parser.add_argument("-k", "--kappa", dest="kappa",
                        help="kappa parameter",
                        metavar="FLOAT", default=2.0)
    parser.add_argument("-b", "--beta_max", dest="beta_max",
                        help="beta max parameter",
                        metavar="FLOAT", default=1e5)


    args = parser.parse_args()

    main(args.imgdir, 
        args.outdir, 
        float(args.lamdaval), 
        float(args.kappa), 
        float(args.beta_max))



