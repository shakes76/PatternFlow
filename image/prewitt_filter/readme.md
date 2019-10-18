# Prewitt Operator


 Tensorflow implementation of the prewitt operator and accompanying short demonstration.

 The prewitt operator performs gradient approximation through a combination of convolutional filters. The gradient approximation of both horizontal and vertical components is used to approximate the gradient of the image intensity. The resultant approximation serves as a simple edge detection algorithm. Input image must be a 2-Dimensional single channel grayscale image.

## Usage

``` 
prewitt_filter(image)
```
 
 >image :: 2D Tensor [width,height] <float32\> representing the grayscale image to apply the filter to.


## Output

Image below shows the original image before the prewittt operator.

![GitHub Logo](./resources/office.jpg)

Image below shows the result of the prewitt operator applied to the above image.

![GitHub Logo](./resources/filtered.jpg)
