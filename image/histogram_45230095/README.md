# Histogram using tensorflow
Basically this function will help to plot a histogram for a grey image. If the image is colored and has multiple channels, the user should separate the image into different channels and then perform the function. <br/>
Suppose the shape of one grey image is (500, 500), the value for each pixel varies from 0 to 255, which means each pixel has 256 possibilities. The histogram function will return the distribution of each value, telling the user how the color varies in a directed way.

## An example of its use
The original image: built-in image Camera <br/>
![alt text](https://raw.githubusercontent.com/LeoYez/PatternFlow/topic-algorithms/image/histogram/camera.png)<br/>
After using the function, the user can plot the histogram as follows:<br/>
![alt text](https://raw.githubusercontent.com/LeoYez/PatternFlow/topic-algorithms/image/histogram/histogram.png)<br/>
From the histogram the user can tell how colors distribute in a more intuitive way.
