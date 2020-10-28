# Improved UNet for ISIC2018 Segmentation



# Description



# How it Works



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

![resoures](1)

<img src="./resoures/1" style="width:90%;" alt="Tesla Trading History" />

# Reference

- [1] <a href="https://www.investopedia.com/articles/basics/04/100804.asp" target="_blank">Forces That Move Stock Prices</a>

[wikipedia - total variation denoising](https://en.wikipedia.org/wiki/Total_variation_denoising)


[scikit-image - denoise bregman](https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_bregman)


[Tom Goldstein and Stanley Osher, “The Split Bregman Method For L1 Regularized Problems”](ftp://ftp.math.ucla.edu/pub/camreport/cam08-29.pdf)




