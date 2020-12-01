# Contrast Limited Adaptive Histogram Equalization (CLAHE)

Tensorflow implementation of the sci-kitequalize_adapthist method in the skimage.exposure module

CLAHE is used to amplify contrast in images to enhance the difference between areas and regions, specificially restricting the contrast by limiting the amplification, thus reducing the amplification of noise. The histogram produced is clipped and then the clipped parts are redistributed across the unclipped components, keeping the total samples.

https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/color/adapt_rgb.py#L35

http://www.realtimerendering.com/resources/GraphicsGems/


