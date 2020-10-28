# Improved UNet for ISIC2018 Segmentation

Improved UNet model of image segmentation implemented for Tensorflow

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

![result](resources/1.png)



# Reference

* [1] [Fabian Isensee et al., "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge"](https://arxiv.org/pdf/1802.10508v1.pdf)

* [2] [Wikipedia - Sørensen Dice coefficient](https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient)

* [3] [Dataset - SICI2018 Challenge](https://challenge2018.isic-archive.com)





