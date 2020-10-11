#!/usr/bin/env python3
from os.path import abspath, dirname
from sys import path
import numpy as np
import tensorflow as tf

path.append(dirname(dirname(abspath(__file__))))

import wiener

def test_astronaut():
    psf = np.ones((5, 5)) / 25
    image_noise = np.load(dirname(abspath(__file__))+'/astronaut_noise.npy')
    image_desired = np.load(dirname(abspath(__file__))+'/astronaut.npy')
    deconvolved = wiener.wiener(image_noise, psf, 1)
    np.testing.assert_allclose(deconvolved, image_desired, rtol=1e-3)
    deconvolved = wiener.wiener(image_noise, psf, 1, is_real=False)
    np.testing.assert_allclose(np.real(deconvolved), image_desired, rtol=1e-3, atol=1e-3)

def test_ir2tf():
    sess = tf.InteractiveSession()
    np.testing.assert_allclose(np.array([[4, 0], [0, 0]]), wiener._ir2tf(np.ones((2, 2)), (2, 2), sess).eval())
    np.testing.assert_allclose((512, 257), (tuple)(tf.shape(wiener._ir2tf(np.ones((2, 2)), (512, 512), sess)).eval()))
    np.testing.assert_allclose((512, 512), (tuple)(tf.shape(wiener._ir2tf(np.ones((2, 2)), (512, 512), sess, is_real=False)).eval()))

def test_laplacian():
    sess = tf.InteractiveSession()
    trans, ir = wiener._laplacian(2, (32, 32), sess)
    np.testing.assert_allclose(np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]), ir.eval())
    np.testing.assert_allclose(trans.eval(), wiener._ir2tf(ir, (32, 32), sess).eval())

if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()