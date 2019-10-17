# L0 Norm Gradient Smoothing 
The L0 Norm Gradient Smoothing algorithm takes images to produce a smoother version of the original image, depending on certain parameters. This is useful for computer vision applications in image enhacement and denoising and 3D mesh denoising. 


## How it works 
L0 gradient smoothing is applied to an input signal (or image in this module) to control non-zero gradients by reducing signal noise by minimizing small gradients and preserving important signal features by retaining larger gradients. 

The following equation is the objective function for the algorithm: 

`F = minimise(S)||S − I||² + λ||∇S||ₒ`

where `||.||²` denotes the L2 norm and `||.||ₒ` indicates the L0 norm and `λ` controls the level of coarseness of the signal, where a larger `λ` produces a less distinguished image. 

This can be seen below with varying values for `λ`:

![Dahlia_output](example/0_dahlia_out_l0.5.png) 
![Dahlia_output](example/1_dahlia_out_l0.2.png) 
![Dahlia_output](example/2_dahlia_out_l0.1.png) 
![Dahlia_output](example/3_dahlia_out_l0.05.png) 
![Dahlia_output](example/4_dahlia_out_l0.02.png) 
![Dahlia_output](example/5_dahlia_out_l0.002.png)
![Dahlia_output](example/6_dahlia_out_l0.0025.png)
![Dahlia_output](example/7_dahlia_out_l0.0002.png)

From left to right, top to bottom `λ = 0.5, 0.2, 0.1, 0.05, 0.02, 0.002, 0.0025, 0.0002` respectively. Note how the dahlia becomes more an more refined as the value of `λ` becomes smaller. 

# Usage 
## Dependencies 
The algorithm is implemented in python version 3. The following python libraries are required to run the module: 

* `tensorflow 2.0` version 2.0 is required for fourier transform calculations. 
* `numpy` for I\O uses, to load and save arrays to and from images 
* `PIL` the image library, for image loading and saving 

## How to run
The script may be run through commandline as such for example:

```
python3 L0_gradient_smoothing_tf.py -d example/dahlia.png -o example/out.png  -l 2e-3 -k 2.0 -b 10000
```

The commandline arguments are described as such: 
```
  -h, --help            show this help message and exit
  -d FILE, --inputimgdir FILE
                        Directory path for input image
  -o FILE, --outdir FILE
                        Directory path for output image
  -l FLOAT, --lamdaval FLOAT
                        lambda parameter
  -k FLOAT, --kappa FLOAT
                        kappa parameter
  -b FLOAT, --beta_max FLOAT
                        beta max parameter
```

Below is the input `dahlia.png` for the example usage (left) and the output of the L0 Norm Gradient Smoothing algorithm (right): 

![Dahlia](example/dahlia_smol.png) 
![Dahlia_output](example/dahlia_out_smol.png)

Otherwise, the method `l0_calc` (found in `L0_gradient_smoothing_tf.py`) takes in a numpy array of the loaded image to compute the smoothed image output array, as well as the relevant parameters: 

* `image_array` loaded image array
* `lambda` determines how 'fine' the smoothing is. Smaller values of lambda give a more detailed image
* `kappa` multiplying factor for the initial beta value, used determine the number of iterations in combination with `beta_max`. 
* `beta_max` max value for beta to reach before reaching the end of the algorithm. 
