---
output:
  html_document: default
  pdf_document: default
---
# Transform module 

match_histograms function implementation in Tensorflow


## Description of the algorithm
In image processing, histogram matching is the transformation of an image so that its  cumulative histogrammatches a specified histogram(from an anohter image). Specifically,algorithmthe Given two images, the reference and the target images, we compute their histograms. Following, we calculate the cumulative distribution functions of the two images' histograms – F1()for the reference image and F2() for the target image. Then for each gray level G1 in [0,255], we find the gray level G2, for which F1(G1) = F2(G2), and this is the result of histogram matching function: M(G1) = G2 Finally, we apply the function M() on each pixel of the reference image.

## How it works
Specifically, firstly, after giving the reference image and the source image. By calculating the cumulative distribution of the reference figure, the probability distribution of each pixel value of the figure can be obtained. Secondly, for each channel, give the information of the reference figure, and use the interpolation function to get the value corresponding to a series of source pictures. Finally, the results of each channel are reconstructed into a figure to achieve the effect of color migration. In addition, You can run run main.py directly to see the effect of the image histogram matching. If you want to use your own image, you need to replace the reference and source in main.py with the path to your image. If you want to see how the specific function is written, you can open match_histograms.py to view it.

## Example 
```python
reference = data.coffee()
source = data.astronaut()
image1  = tf.convert_to_tensor(image1)
image2 = tf.convert_to_tensor(image2)
matched = mh.match_histograms(image1, image2, multichannel=True)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)
for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

ax1.imshow(image1)
ax1.set_title('Source')
ax2.imshow(image2)
ax2.set_title('Reference')
ax3.imshow(matched)
ax3.set_title('Matched')
plt.tight_layout()
plt.show()
```

The example of completing color migration of astronaut image


![example](./img-storage/Example.png)

## Dependencies

This project is tested in following context:

- macOS Catalina 10.15
- python 3.7.3 64-bit('base": conda)
- Tensorflow version: 1.14.0

The implementation of the entire function is done in the eager_execution mode of tensorflow. If your tensorflow version does not support this feature, please update the tensorflow version to 1.14.0 or more.


Author: Tianjie Shi
Last update: 16/10/2019
