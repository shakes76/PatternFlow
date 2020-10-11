# Denoising - denoise_tv_chambolle
Denoise algorthm implemented for Pytorch

## Description
In signal processing , total variation denoising, also know as total variation regularization is a process often used in digital image processing. It has applications in noise removal. However, many packages does not support the data type tensor. This function implement the denoise_tv_chambolle of skimage in torch.tensor format. The code and the testing results are shown in this repository. 

## Environment 
* Python version: 3.7
* torch version : 1.2.0
* mac os version: 10.14.6
* test cast: 
[test case from Skimage](https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/restoration/tests/test_denoise.py#L213)

## Algorithm
* __Parameters:__
	1. __input_img : torch.tensor__ 
	
		n-D input data to be denoised.
				
	2. __weight : float optional__ 
	
		Denoising weight. The greater 'weight', the more denoising (aT the expense of fidelity to 'input').
			
	3. __eps : float optional__
		
		Relative difference of the value of the cost function that determines the stop criterion. The algorithm stops when:
	(E_(n-1) - E_n) < eps * E_0
			
	4. __n_iter_max : int optional__
	
		Maximal number of iterations used for the optimization.
	5. __multichannel : bool optional__
	
		Apply total-variation denoising separately for each channel. This option should be true for color images, otherwise the denoising is also applied in the channels dimension.
			
* __Returns:__

	1. __out : torch.tensor__
	
		Denoised image.

## Example 
* Comparisons between the original image and denoised image in torch and numpy
```
    from skimage import data, img_as_float, restoration
    import torch
    import matplotlib.pyplot as plt
  
    coffee = img_as_float(data.coffee())    
    #add noise to the original image to test the denoising effect
    noise  = torch.randn(coffeeT.size(),dtype = coffeeT.dtype)*0.15
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(8,8))
    ax1.imshow(coffee+noise.numpy())
    ax1.set_title('Original Image')
    #result of torch
    ax2.imshow(denoise_tv_chambolle_torch(coffeeT+noise))
    ax2.set_title('Denoised(torch)')
    #result of numpy
    ax3.imshow(restoration.denoise_tv_chambolle(coffee+noise.numpy()))
    ax3.set_title('Denoised(numpy)')
    plt.show()     
```
## Results of denoised image

![result](https://i.imgur.com/zEvRThr.png)


## Author
Name : Weichung Lai
Student No. : 45033027