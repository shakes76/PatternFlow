Transform module 
====
match_histograms function implementation in Tensorflow
-------
# Description of the algorithm
The function of this algorithm is: Adjust an image so that its cumulative histogram matches that of another iamge. Specifically, firstly, after giving the reference image and the source image. By calculating the cumulative distribution of the reference figure, the probability distribution of each pixel value of the figure can be obtained. Secondly, for each channel, give the information of the reference figure, and use the interpolation function to get the value corresponding to a series of source pictures. Finally, the results of each channel are reconstructed into a figure to achieve the effect of color migration.

# How it works
You can run run main.py directly to see the effect of the image color style migration. If you want to use your own image, you need to replace the reference and source in main.py with the path to your image. If you want to see how the specific function is written, you can open match_histograms.py to view it.

## Example 
The example of completing color migration of astronaut image
