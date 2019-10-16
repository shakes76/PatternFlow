## load modules
import tensorflow as tf
import math
# In[2]:

def iradon(radon_image, theta=None, output_size=None, filter="ramp", interpolation="linear", circle=True):
    sess = tf.InteractiveSession()
    radon_image = tf.constant(radon_image, tf.float64)
    
    if tf.rank(radon_image).eval() != 2:
        print(tf.rank(radon_image))
        raise ValueError('The input image must be 2-D')
   
    if theta is None:
        theta = tf.linspace(0.0, 180.0, tf.shape(radon_image)[1])
    
    angles_count = tf.shape(theta)[0]
    
    if angles_count.eval() != tf.shape(radon_image)[1].eval():
        raise ValueError("The given ``theta`` does not match the number of projections in ``radon_image``.")
    
    interpolation_types = ('linear', 'nearest', 'cubic')
    if interpolation not in interpolation_types:
        raise ValueError("Unknown interpolation: %s" % interpolation)
  
    filter_types = ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None)
    if filter not in filter_types:
        raise ValueError("Unknown filter: %s" % filter)
  
    img_shape = tf.shape(radon_image)[0].eval()
    if output_size is None:
    # If output size not specified, estimate from input radon image
        if circle:
            output_size = img_shape
        else:
            output_size = int(math.floor(math.sqrt((img_shape) ** 2 / 2.0)))
    
    if circle:
        radon_image = _sinogram_circle_to_square(radon_image)
        img_shape = radon_image.shape[0]
        
    # Resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = max(64, int(2 ** tf.math.ceil(log2(2 * img_shape)).eval()))
    pad_width = ((0, projection_size_padded - img_shape), (0, 0))
    img = tf.pad(radon_image, pad_width, mode='CONSTANT', constant_values=0)
    
    # Apply filter in Fourier domain
    fourier_filter = _get_fourier_filter(projection_size_padded, filter)
    complex_img = tf.dtypes.cast(img, tf.complex64)
    projection = tf.signal.fft(complex_img) * tf.constant(fourier_filter, tf.complex64)
    radon_filtered = tf.math.real(tf.signal.ifft(projection)[:img_shape, :])
    _debug_plot(img.eval(), radon_filtered.eval())
    print(radon_filtered.eval())
    
    sess.close()
    
# In[3]:
def _sinogram_circle_to_square(sinogram):
#    diagonal = int(np.ceil(np.sqrt(2) * sinogram.shape[0]))
#    pad = diagonal - sinogram.shape[0]
#    old_center = sinogram.shape[0] // 2
#    new_center = diagonal // 2
#    pad_before = new_center - old_center
#    pad_width = ((pad_before, pad - pad_before), (0, 0))
#    return np.pad(sinogram, pad_width, mode='constant', constant_values=0)
    pass

def _get_fourier_filter(size, filter_name):
    import numpy as np
    import numpy.fft
    fftmodule = numpy.fft
    fft = fftmodule.fft
    ifft = fftmodule.ifft
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=np.int),
                        np.arange(size / 2 - 1, 0, -2, dtype=np.int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    # Computing the ramp filter from the fourier transform of its
    # frequency domain representation lessens artifacts and removes a
    # small bias as explained in [1], Chap 3. Equation 61
    fourier_filter = 2 * np.real(fft(f))         # ramp filter
    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = np.pi * fftmodule.fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = fftmodule.fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= fftmodule.fftshift(np.hamming(size))
    elif filter_name == "hann":
        fourier_filter *= fftmodule.fftshift(np.hanning(size))
    elif filter_name is None:
        fourier_filter[:] = 1

    return fourier_filter[:, np.newaxis]
    

# In[4]:
def log2(x):
    n = tf.math.log(tf.constant(x, tf.float64))
    d = tf.math.log(tf.constant(2, dtype=n.dtype))
    return n / d
# In[4]:
def _debug_plot(original, result, sinogram=None):
    from matplotlib import pyplot as plt
    imkwargs = dict(cmap='gray', interpolation='nearest')
    if sinogram is None:
        plt.figure(figsize=(15, 6))
        sp = 130
    else:
        plt.figure(figsize=(11, 11))
        sp = 221
        plt.subplot(sp + 0)
        plt.imshow(sinogram, aspect='auto', **imkwargs)
    plt.subplot(sp + 1)
    plt.imshow(original, **imkwargs)
    plt.subplot(sp + 2)
    plt.imshow(result, vmin=original.min(), vmax=original.max(), **imkwargs)
    plt.subplot(sp + 3)
    plt.imshow(result - original, **imkwargs)
    plt.colorbar()
    plt.show()









