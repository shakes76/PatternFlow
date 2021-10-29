# Total-Variation Denoising using split-Bregman optimization

Total-variation denoising algorthm using split-Bregman optimization implemented for Pytorch

# Description

Total-variation denoising is a process, most often used in digital image processing, that has applications in noise removal. It is based on the principle that signals with excessive and possibly spurious detail have high total variation, that is, the integral of the absolute gradient of the signal is high. According to this principle, reducing the total variation of the signal subject to it being a close match to the original signal, removes unwanted detail whilst preserving important details such as edges. Total variation denoising is remarkably effective at simultaneously preserving edges whilst smoothing away noise in flat regions, even at low signal-to-noise ratios.

# How it Works

Total-variation denoising is proposed to estimate the denoised image u as the solution to the following minimise optimisation problem

![](objective_function.png)

, where lambda is a positive parameter and f is the noisy image.

The objective function is designed to minimise the total variation of the denoised image while keeping the differnece between denoised image and noisy image small, making the denoised image more alike the noisy image.

The split-Bregman method is a optimisation technique for solving non-differentiable convex minimization problems, and it is especially efficient for problems with L1 or TV regularization.


# Dependencies

__Requirement:__

* Python 3.6
* Pytorch 1.2.0

[Download]() to use the algorithm

or via Github clone

```shell
git clone https://github.com/1tanwang/PatternFlow.git
```


# Example Usage

Import the algorithm and use in Python script.

__Note: The input image to algorithm should be converted into pytorch tensor__

```Python
import torch

from PatternFlow.denoise.denoise_tv_bregman.denoise_tv_bregman import denoise_tv_bregman

# noisy_img is the noised input image
# convert to torch tensor
input_img = torch.FloatTensor(noisy_img)
# pass the input image to algorithm
# convert the return image to numpy array for later plotting
denoised_img = denoise_tv_bregman(input_img, weight=0.1).numpy()
```

# Example Result

The original image in following example result comes from scikit-image. The noisy image is original image added with Guassian noise. And the denoised image is the output image from the total-variation denoising bregman algorithm.

![](cat.png)

# Reference

[wikipedia - total variation denoising](https://en.wikipedia.org/wiki/Total_variation_denoising)

[scikit-image - denoise bregman](https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_bregman)

[Tom Goldstein and Stanley Osher, “The Split Bregman Method For L1 Regularized Problems”](ftp://ftp.math.ucla.edu/pub/camreport/cam08-29.pdf)

[Pascal Getreuer, “Rudin–Osher–Fatemi Total Variation Denoising using Split Bregman”](https://www.ipol.im/pub/art/2012/g-tvd/article_lr.pdf)
