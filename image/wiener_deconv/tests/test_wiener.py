#!/usr/bin/env python3
from os.path import abspath, dirname, join as pjoin
import sys
import numpy as np
from skimage import color, data

sys.path.append(dirname(dirname(abspath(__file__))))

import wiener

def test_astronaut():
    psf = np.ones((5, 5)) / 25
    deconvolved = wiener.wiener(np.load('astronaut_nosiy.npy'), psf, 1)
    np.testing.assert_allclose(deconvolved, np.load('astronaut.npy'), rtol=1e-3)
    
def test_astronaut_real():
    psf = np.ones((5, 5)) / 25
    deconvolved = wiener.wiener(np.load('astronaut_nosiy.npy'), psf, 1, is_real=False)
    np.testing.assert_allclose(np.real(deconvolved), np.load('astronaut.npy'), rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    test_astronaut()
    test_astronaut_real()