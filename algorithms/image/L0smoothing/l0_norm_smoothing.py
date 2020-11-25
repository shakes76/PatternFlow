# L0 Gradient Image Norm Algorithm
#
# author: naziah siddique (44315203)
#
# References:
# [1]   Xu, L., Lu, C., Xu, Y., & Jia, J. (2011, December).
#       Image smoothing via L 0 gradient minimization.
#       In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p. 174). ACM.


import tensorflow as tf

# Require tensorflow version >=2.0 to run
print(tf.__version__)


def psf2otf(psf, outSize):
    """
    Adapted from the original paper MATLAB implementation. [1]
    Returns a padded matrix of outSize, with values based on psf

    Args:
        psf: either [[1,-1]] or [[1],[-1]] depending on x, y
        outSize: shape

    Returns:
        padded tensor of outSize, with centred psf
    """

    psfSize = tf.constant(psf.shape)

    new_psf = tf.Variable(tf.zeros(outSize, dtype=tf.float32))
    
    new_psf = new_psf[:psfSize[0],:psfSize[1]].assign(psf)
    psf = new_psf
    
    shift = -(psfSize / 2)
    for i in range(shift.shape[0]):
        psf = tf.roll(psf, int(shift[i]), axis=i)
    
    otf = tf.signal.fft2d(tf.complex(psf, tf.constant(tf.zeros(psf.shape))))

    return otf


def l0_calc(img_arr, _lambda=2e-2, kappa=2.0, beta_max=1e5):
    """
    Args:
        img_arr: input signal (image in this case)
        _lambda: controls the level of coarseness of the input signal
        kappa: controls the rate. Smaller kappa gives more iterations and images with sharper edges
        beta_max: controls the number of iterations too by determining the interval of convergence

    Returns:
         output array of the L0 gradient norm of img_arr
    """
    (N, M, D) = img_arr.shape

    # Initialise and normalise S with image I 
    S = tf.Variable(img_arr, dtype=tf.float32)
    S = S.assign(S/256)

    # define fx, and fy 
    size_2D = (N, M)
    fx = tf.constant([[1, -1]], dtype=tf.float32)
    fy = tf.constant([[1], [-1]], dtype=tf.float32)

    otfFx = psf2otf(fx, size_2D)
    otfFy = psf2otf(fy, size_2D)

    S_complex = tf.Variable(tf.complex(S, tf.constant(tf.zeros((N, M, D)))))
    S_complex = tf.cast(S_complex, dtype=tf.complex128)
    S_complex = tf.Variable(S_complex)

    # initialise FI and assign fourier transforms on S 
    FI = tf.Variable(tf.complex(tf.zeros((N, M, D)), tf.zeros((N, M, D))))
    FI = tf.cast(FI, dtype=tf.complex128)
    FI = tf.Variable(FI)
    for i in range(S_complex.shape[2]):
        FI = FI[:,:,i].assign(tf.cast(tf.signal.fft2d(S_complex[:,:,i]), dtype=tf.complex128))

    # get sum of the gradient magnitudes 
    MTF = tf.math.square(tf.math.abs(otfFx)) + tf.math.square(tf.math.abs(otfFy))
    MTF = tf.stack([MTF, MTF, MTF], axis=2)

    # initialise x, y gradient matrices, dx, dy 
    h = tf.Variable(tf.zeros((N, M, D), dtype=tf.float64))
    v = tf.Variable(tf.zeros((N, M, D), dtype=tf.float64))
    dxhp = tf.Variable(tf.zeros((N, M, D), dtype=tf.float64))
    dyvp = tf.Variable(tf.zeros((N, M, D), dtype=tf.float64))
    FS = tf.Variable(tf.complex(tf.zeros((N, M, D), dtype=tf.float64), 
                                tf.zeros((N, M, D), dtype=tf.float64) ))
    FS = tf.cast(FS, dtype=tf.complex128)

    beta = 2 * _lambda  # set initial beta value
    iterations = 0               # used to track number of iterations 

    # iterate until convergence 
    while beta < beta_max:

        # initialise and calculate h,v
        h = tf.Variable(h)
        v = tf.Variable(v)

        h = h[:,0:M-1,:].assign(
            tf.cast(tf.math.real(S_complex[:,1:]-S_complex[:,:-1]), dtype=tf.float64))
        h = h[:,M-1:M,:].assign(
            tf.cast(tf.math.real(S_complex[:,0:1,:]-S_complex[:,M-1:M,:]), dtype=tf.float64))

        v = v[0:N-1,:,:].assign(
            tf.cast(tf.math.real(S_complex[1:]-S_complex[:-1]), dtype=tf.float64))
        v = v[N-1:N,:,:].assign(
            tf.cast(tf.math.real(S_complex[0:1,:,:] - S_complex[N-1:N,:,:]), dtype=tf.float64))
        
        # mask and update h,v
        t = tf.reduce_sum(
            tf.math.square(h) + tf.math.square(v), axis=2) < _lambda / beta
        t = tf.stack([t, t, t], axis=2)
        idx = tf.where(t)
        h = tf.tensor_scatter_nd_update(h, idx, tf.zeros(idx.shape[0], dtype=tf.float64))
        v = tf.tensor_scatter_nd_update(v, idx, tf.zeros(idx.shape[0], dtype=tf.float64))

        # calculate dx, dy 
        dxhp = dxhp[:,0:1,:].assign(h[:,M-1:M,:] - h[:,0:1,:])
        dxhp = dxhp[:,1:M,:].assign(-(h[:,1:]-h[:,:-1]))
        dyvp = dyvp[0:1,:,:].assign(v[N-1:N,:,:] - v[0:1,:,:])
        dyvp = dyvp[1:N,:,:].assign(-(v[1:]-v[:-1]))

        # calculate and cast normin to complex to feed into fourier transform 
        normin = dxhp + dyvp
        normin = tf.complex(
            normin, tf.constant(tf.zeros(normin.shape, dtype=tf.float64)))

        # fourier transform for each colour channel in S 
        for i in range(S.shape[2]):
            FS = FS[:,:,i].assign(tf.signal.fft2d(normin[:,:,i]))
        
        # calculate denorm 
        denorm = tf.cast(1 + beta * MTF, dtype=tf.complex128)
        FS = FS[:,:,:].assign((FI + beta * FS) / denorm)
        
        # compute inverse fourier transform 
        for i in range(S.shape[2]):
            S_complex = tf.Variable(S_complex) 
            S_complex = S_complex[:,:,i].assign(tf.signal.ifft2d(FS[:,:,i]))

        beta *= kappa           # increase beta value kappa times 
        iterations += 1         # keep track of iterations made 

    # Rescale image
    S_complex = S_complex * 256

    print("Iterations made: %d" % (iterations))

    # convert real S values to array
    output_arr = tf.math.real(S_complex).numpy()

    return output_arr


