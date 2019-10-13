# Tensorflow impletmention of skimage.exposure algorithms

## Depends:
* Tensorflow
* Tensorflow_probability
* matplotlib

## Usage
Example usage can but found in histogram_mertics_driver_scripte.py.
Note all funcations return either nothing (for plotting funcations) or a list of tensors that excute the required computation (for image_histogram and cumulative_distribution) or a singal tensor (for equalize_hist_by_index and equalize_hist_by_image).
These tensors need to be evulated just like other tensorflow tensors in order to get the data.
### Quickstart guide
* pass a list of pictures to the histogram_mertics object
* you can now call the funcations plot_histogram and plot_cdf to plot the images colour channel historgram and cdf
* you can equalize an image passed in in the init if you know the index via equalize_hist_by_index(index)
* you can equalize any other image by passing it in via equalize_hist_by_image

Note all following examples can be recreated by simply runnning histogram_mertics_driver_script.py

## histogram

For each colour channel in all the images passed in the image histgram is calculated. The funcation doing most of the work is the inbuilt tensorflow funcation tf.histogram_fixed_width.
Example run on cifar10 data
![histogram example](https://i.imgur.com/37RaCVq.png)

## cumulative distribution

For each colour channel the cummulative distributions of that colours histgramm can be calculated rather simiply with tf.math.cumsum and normalizition
Example run on cifar10 data
![Cummlative distrubution example](https://i.imgur.com/E2dCTJK.png)

## equalize hist

in order to equalize images a funcation from tensorflow probability needs to be used. tfp.amth.interp_regular_1d_grid always the interpational needed to equalize the image with respect to the cumulative distribution

Example run on images passed in during init:

![Equalize hist by index before](https://i.imgur.com/NjCblSV.png)

![Equalize hist by index after](https://i.imgur.com/2n364ZI.png)

Example on passing in new images (note new dims are from needed padding):

![Equalize hist on new image before](https://i.imgur.com/FbS9sBR.png)

![Equalize hist on new image after](https://i.imgur.com/48RaVme.png)