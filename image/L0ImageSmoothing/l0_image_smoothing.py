from imageio import imread
import tensorflow as tf


def _pad_fxy_psf(psf, shape):
    """
    Pads point spread function (psf) Fx or Fy with zeroes up to target shape.
    The target shape is the shape of the image we want to smooth.

    This method is a pre-processing step prior to conversion to an optical
    transfer function (OTF).

    :param psf: Tensor containing a point spread function for padding.
    :param shape tuple(int, int): Target shape from image we are smoothing.
    :return: psf function padded with zeroes with new shape=(shape[0], shape[1])
    """
    if psf.shape[0] == 1:
        # PSF is Fx = [[-1, 1]]
        indices = [[0, 0], [0, 1]]
        psf_padded = tf.SparseTensor(indices, psf[0], shape)
        psf_padded = tf.sparse.to_dense(psf_padded, default_value=0)
    elif psf.shape[0] == 2:
        # PSF is Fy = [[-1], [1]]
        indices = [[0, 0], [1, 0]]
        psf_padded = tf.SparseTensor(indices, psf[:, 0], shape)
        psf_padded = tf.sparse.to_dense(psf_padded, default_value=0)
    return psf_padded

def _fxy_psf2otf(psf, target):
    """

    :return:
    """
    target_shape = (target.shape[0], target.shape[1])
    # Zero Pad template, we want psf to be padded first with zeroes.
    psf_padded = _pad_fxy_psf(psf, target_shape)

    return psf_padded


def l0_image_smoothing():
    """

    :return:
    """
    return

if __name__ == '__main__':
    img = imread('D:\Code\comp3710\L0smoothing\code\pflower.jpg')
    print(f'img shape: {img.shape}')
    psf = tf.constant([[-1, 1]], tf.int8)
    otf = _fxy_psf2otf(psf, img)
    psf2 = tf.constant([[-1], [1]], tf.int8)
    otf2 = _fxy_psf2otf(psf2, img)
    print(otf.numpy(), otf.shape)
    print(otf2.numpy(), otf2.shape)
