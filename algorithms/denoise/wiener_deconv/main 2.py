from skimage import data
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import numpy as np
from wiener import wiener

def get_imgs(psf):
    img = data.camera()
    img_noise = convolve2d(img, psf, 'same')
    img_noise += 0.5 * img_noise.std() * np.random.standard_normal(img_noise.shape)
    return img_noise, img

def main():
    psf = np.ones((5, 5)) / 25
    #get the noised image
    img_noise, origin = get_imgs(psf)
    fig = plt.figure(figsize=(10, 5))
    #apply wiener deconvolution to noised image
    img_denoised = wiener(img_noise, psf, 0.7)
    #show result and comparison
    fig.add_subplot(1,3,1)
    plt.title("noised")
    plt.imshow(img_noise)
    fig.add_subplot(1,3,2)
    plt.title("denoised")
    plt.imshow(img_denoised)
    fig.add_subplot(1,3,3)
    plt.title("original")
    plt.imshow(origin)
    #plt.show()
    plt.savefig('camera.png')
    
if __name__ == '__main__':
    main()