**Sobel Transform in tensorflow**
Based on skimage.filters.sobel(image)

**Demo**
Tensorflow implementation of sobel transform. 
It outputs the edge map of the given image. 
Given below is a picture of katy perry and the corresponding edge detection image.



**Sobel Transform**

Sobel Transform is a edge detection algorithm used mainly in the fields of computer vision and image processing.
It creates an edge map from it's corresponding input image.
It is based on convolution of the image with the appropriate kernels (called Sobel kernels), to give the resultant edge maps.

**Kernels used**
Here we have used 2 kernels, one for horizontal edge detection and the other for vertical edge detection.

**Sobel_h (detects horizontal edges) = [ [1,2,1], [0,0,0], [-1,-2,-1] ] ** 

**Sobel_v (detects vertical edges) = [ [1,0,-1], [2,0,-2], [1,0,-1] ] ** 

