# Pixel-Based Visual Information Fidelity (PB-VIF)
_Author: Antoine DELPLACE_  
_Last update: 30/09/2019_

Visual Infomation Fidelity (VIF) is a measure that evaluates the relative quality of a compressed or altered image compared to the original figure. It uses the Human Visual System (HVS) representation and a Gaussian Scale Mixture (GSM) model in the wavelet domain to create the index (see Method Description).

## Usage
### Dependencies
- Python 3.6
- Tensorflow 1.13 -- `vif.py`
- Numpy 1.16 -- `main.py`
- Imageio 2.5 -- `main.py`
- Tabulate 0.8.5 -- `main.py`

### Built-in driver `main.py`
The `main()` function imports the images, performs a grayscale conversion, calls the function `pbvif()` of the `vif` module and returns a formatted table with the results.
```sh
main.py reference_image_filename query_image_filename1 query_image_filename2 query_image_filename3 ...
```

### Function pbvif in `vif.py`
```
pbvif(ref, query_tab, max_scale=4, var_noise=2.0, mode='nearest')
    Computes the Pixel-Based Visual Information Fidelity (PB-VIF) using Tensorflow

    Parameters
    ----------
    ref       : 2-dimension grayscaled image reference.
    query_tab : list containing the 2-dimension grayscaled images to be compared with.
    max_scale : Number of subbands to extract information (Default: 4)
    var_noise : Variance of additive noise (HVS model parameter, Default: 2.0)
    mode      : mode used for padding convolutions (Default: "nearest")
        - "nearest"   : the input is extended by replicating the last pixel
        - "symmetric" : the input is extended by reflecting about the edge of the last pixel
        - "constant"  : the input is extended by filling all values beyond the edge with zeros

    Return
    ----------
    Pixel-Based Visual Information Fidelity (float between 0 and 1)
```

### Example
```sh
main.py lena_ref.bmp lena_gauss_noise.bmp lena_low_pass_filter.bmp lena_jpeg.jpg lena_ref.bmp
```

| ![lena_gauss_noise.bmp](lena_gauss_noise.bmp) | ![lena_low_pass_filter.bmp](lena_low_pass_filter.bmp) | ![lena_jpeg.jpg](lena_jpeg.jpg) | ![lena_ref.bmp](lena_ref.bmp) |  
|:--:|:--:|:--:|:--:|  
| *Image with Gaussian noise* | *Low-pass filtered image* | *JPEG image* | *Reference image* |  

```sh
| Image                    |    pbvif |
|--------------------------|----------|
| lena_gauss_noise.bmp     | 0.226277 |
| lena_low_pass_filter.bmp | 0.385636 |
| lena_jpeg.jpg            | 0.837977 |
| lena_ref.bmp             | 1        |
```

## Method Description
The method uses the following framework. The Source Model is defined with ![https://wikimedia.org/api/rest_v1/media/math/render/svg/f7d7ec2e74a05b3f674f0a78e5d6c9eedc75dcc0](https://wikimedia.org/api/rest_v1/media/math/render/svg/f7d7ec2e74a05b3f674f0a78e5d6c9eedc75dcc0), the wavelet coefficients for a given subband. The Gaussian Scale Mixture model gives the decomposition ![https://wikimedia.org/api/rest_v1/media/math/render/svg/abf22b47efa9b524eb2026014d2d11d3529f90fa](https://wikimedia.org/api/rest_v1/media/math/render/svg/abf22b47efa9b524eb2026014d2d11d3529f90fa) with ![https://wikimedia.org/api/rest_v1/media/math/render/svg/de6e810a93f67802ecb603ee0e3324005c6e583e](https://wikimedia.org/api/rest_v1/media/math/render/svg/de6e810a93f67802ecb603ee0e3324005c6e583e), a positive scalar and ![https://wikimedia.org/api/rest_v1/media/math/render/svg/bee7e772e4aff8f6f7d70dc66e9c340962226b56](https://wikimedia.org/api/rest_v1/media/math/render/svg/bee7e772e4aff8f6f7d70dc66e9c340962226b56), a Gaussian vector with zero mean and co-variance ![https://wikimedia.org/api/rest_v1/media/math/render/svg/ba27fb01ff5e4d5e0069546f8a61e9cd524f6302](https://wikimedia.org/api/rest_v1/media/math/render/svg/ba27fb01ff5e4d5e0069546f8a61e9cd524f6302).  

The Distortion Model defines the distorted image random field ![https://wikimedia.org/api/rest_v1/media/math/render/svg/3ae0ac2c59f8a698e50ce6db302c1320eeea1808](https://wikimedia.org/api/rest_v1/media/math/render/svg/3ae0ac2c59f8a698e50ce6db302c1320eeea1808) as the product of the source by the scalar field ![https://wikimedia.org/api/rest_v1/media/math/render/svg/3a1f9f42f25c949943ff4f05f048960a911be6be](https://wikimedia.org/api/rest_v1/media/math/render/svg/3a1f9f42f25c949943ff4f05f048960a911be6be) with an additive Gaussian noise ![https://wikimedia.org/api/rest_v1/media/math/render/svg/f6cfc2ec2b582a9d4bedc55e7dc7cfadb5f3eeb1](https://wikimedia.org/api/rest_v1/media/math/render/svg/f6cfc2ec2b582a9d4bedc55e7dc7cfadb5f3eeb1) of mean zero and co-variance ![https://wikimedia.org/api/rest_v1/media/math/render/svg/c0748d9951b30e123da71ae964a09d18253ae3c1](https://wikimedia.org/api/rest_v1/media/math/render/svg/c0748d9951b30e123da71ae964a09d18253ae3c1).  

Finally, the Human Visual System (HVS) model adds a white Gaussian noise to both the source and the distorted images to model visual uncertainty: ![https://wikimedia.org/api/rest_v1/media/math/render/svg/9d6ab317d316a631b045ffca9b0c32f7eb9cf8b0](https://wikimedia.org/api/rest_v1/media/math/render/svg/9d6ab317d316a631b045ffca9b0c32f7eb9cf8b0) and ![https://wikimedia.org/api/rest_v1/media/math/render/svg/e4a9b87bbe0c90a1b265de7f0edd49c7e3a389bf](https://wikimedia.org/api/rest_v1/media/math/render/svg/e4a9b87bbe0c90a1b265de7f0edd49c7e3a389bf) with ![https://wikimedia.org/api/rest_v1/media/math/render/svg/b7551c7bed2cd2ee83e10536d157c94a5f8f72fd](https://wikimedia.org/api/rest_v1/media/math/render/svg/b7551c7bed2cd2ee83e10536d157c94a5f8f72fd) and ![https://wikimedia.org/api/rest_v1/media/math/render/svg/ac1867ecee3572e90910ef9971cb87343b179a6c](https://wikimedia.org/api/rest_v1/media/math/render/svg/ac1867ecee3572e90910ef9971cb87343b179a6c) of mean zero and co-variance ![https://wikimedia.org/api/rest_v1/media/math/render/svg/a277d5d0c956d1e8da7d99a2f973bb48f2f082ff](https://wikimedia.org/api/rest_v1/media/math/render/svg/a277d5d0c956d1e8da7d99a2f973bb48f2f082ff).  

The VIF index can then be computed using ![https://wikimedia.org/api/rest_v1/media/math/render/svg/5786d7409aeed548a519f9d857f04c78f1977203](https://wikimedia.org/api/rest_v1/media/math/render/svg/5786d7409aeed548a519f9d857f04c78f1977203) as the maximum likelihood estimate of ![https://wikimedia.org/api/rest_v1/media/math/render/svg/a1e9b324b2b9bb76e58fd0dc2cf9db713a8d647f](https://wikimedia.org/api/rest_v1/media/math/render/svg/a1e9b324b2b9bb76e58fd0dc2cf9db713a8d647f) given ![https://wikimedia.org/api/rest_v1/media/math/render/svg/cd8ba777375ec9e42e7bd6d8671da387e48e2936](https://wikimedia.org/api/rest_v1/media/math/render/svg/cd8ba777375ec9e42e7bd6d8671da387e48e2936) and ![https://wikimedia.org/api/rest_v1/media/math/render/svg/ba27fb01ff5e4d5e0069546f8a61e9cd524f6302](https://wikimedia.org/api/rest_v1/media/math/render/svg/ba27fb01ff5e4d5e0069546f8a61e9cd524f6302) :

![https://wikimedia.org/api/rest_v1/media/math/render/svg/05fb6e33e22a34562b5ef795544929f929607ba5](https://wikimedia.org/api/rest_v1/media/math/render/svg/05fb6e33e22a34562b5ef795544929f929607ba5)  

![https://wikimedia.org/api/rest_v1/media/math/render/svg/219f20288338fa055890f32301b982ef767fb812](https://wikimedia.org/api/rest_v1/media/math/render/svg/219f20288338fa055890f32301b982ef767fb812)  

![https://wikimedia.org/api/rest_v1/media/math/render/svg/f30e12cffbf4c056f700dd8baccdf57225f3e747](https://wikimedia.org/api/rest_v1/media/math/render/svg/f30e12cffbf4c056f700dd8baccdf57225f3e747)

## References
1. Hamid R. Sheikh and Alan C. Bovik. "Image information and visual quality". In: *IEEE Trans. Image Processing.* 2004, pp. 430–444.
2. Pedro Garcia-Freitas. "PyMetrikz: Python Visual Quality Metrics Package". [Bitbucket repository](https://bitbucket.org/kuraiev/pymetrikz/). 2014
3. Wikipedia. "Visual Information Fidelity". \[[accessed 30 September 2019](https://en.wikipedia.org/wiki/Visual_Information_Fidelity)\]