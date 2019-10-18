if __name__ == "__main__":
    from skimage import color, data, restoration
    import richardson_lucy as rl
    import numpy as np
    import tensorflow as tf
    camera = color.rgb2gray(data.camera())
    from scipy.signal import convolve2d
    import matplotlib.pyplot as plt

    psf = np.ones((5, 5)) / 25
    camera = convolve2d(camera, psf, 'same')
    camera += 0.1 * camera.std() * np.random.standard_normal(camera.shape)
    camera = tf.convert_to_tensor(camera)
    psf = tf.convert_to_tensor(psf)
    deconvolved = rl.richardson_lucy(camera, psf, 5, False)
    plt.imshow(deconvolved)
    plt.show()
