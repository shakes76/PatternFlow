# Tensorflow impletmention of skimage.exposure algorithms
Implements image equalization via histogram, which means the image histogram and cumulative distribution also need to be calculated. 

## Depends:
* Tensorflow
* Tensorflow_probability
* matplotlib

## Usage
Example usage can but found in histogram_mertics_driver_scripte.py.
Note all functions return either nothing (for plotting functions) or a list of tensors that execute the required computation (for image_histogram and cumulative_distribution) or a single tensor (for equalize_hist_by_index and equalize_hist_by_image).
These tensors need to be evaluated just like other tensorflow tensors in order to get the data.
### Quickstart guide
* pass a list of pictures to the histogram_mertics object
* you can now call the functions plot_histogram and plot_cdf to plot the images colour channel histogram and cdf
* you can equalize an image passed in in the init if you know the index via equalize_hist_by_index(index)
* you can equalize any other image by passing it in via equalize_hist_by_image

# Theory

Histogram equalization is a method in increase the global contrast of an image. In a nutshell it works via spreading out the most frequent intensity values. It does this via binning the flattened image data from each colour channel to create the image histogram. This histograms are then used to created a cumulative distribution function (CDF) via a cumulative sum with normalization. This CDF is then used to interperal from the image histogram, the result of which is the equalized image. This has the best effect in images with constant brightness. It is commonly used on scientific imaging such as X-rays, thermal imaging or satellite imaging due to it’s advantage of very low computational overhead, making it useful it you need to equalize images in real time or equalize a large amount of images.

# Funcations

##Driver script
Note all following examples can be recreated by simply running 
python histogram_mertics_driver_script.py

## histogram

For each colour channel in all the images passed in the image histogram is calculated. The function doing most of the work is the inbuilt tensorflow function tf.histogram_fixed_width. The function returns a list of tensors that calculate the result for each colour channel. It also saves the result in the object for later use
Example run on cifar10 data

![histogram example](https://i.imgur.com/37RaCVq.png)

## cumulative distribution

For each colour channel the cumulative distributions of that colours histogram can be calculated rather simply with tf.math.cumsum and normalization. The function returns a list of tensors that calculate the result for each colour channel. It also saves the result in the object for later use
Example run on cifar10 data

![Cumulative distribution example](https://i.imgur.com/E2dCTJK.png)

## equalize hist

in order to equalize images a function from tensorflow probability needs to be used. tfp.amth.interp_regular_1d_grid does the interpolation needed to equalize the image with respect to the cumulative distribution. The function returns a tensor that calculate the result for the chosen image.

Example run on images passed in during init:

![Equalize hist by index before](https://i.imgur.com/NjCblSV.png)

![Equalize hist by index after](https://i.imgur.com/fUejoj7.png)

Example on passing in new images (note new dims are from needed padding):

![Equalize hist on new image before](https://i.imgur.com/FbS9sBR.png)

![Equalize hist on new image after](https://i.imgur.com/zkJEIKo.png)
