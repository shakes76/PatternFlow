# Sobel Transform

*Author*: Vaibhavi Sanjeet Itkyal *Last update*: 16/10/2019

The presented module uses an image from *skimage* dataset, computes and displays it's horizontal, vertical or magnitude of the edges.
The user has a choice to view either horizontal edges, vertical edges or magnitude of the edges of the image.

Currently the program uses an example of *camera* image from *skimage* dataset. 

# Algorithm 
The Sobel operator basically can be used for the edge detection of an image. 

**Input parameter** : Image to process (2-D Array)

**Returns** : Sobel Edge Map (2-D Array)

**Sobel Horizontal Edges**

Horizontal kernel : 

                    1    2    1
                    0    0    0
                   -1   -2   -1

**Sobel Vertical Edges**

Vertical kernel : 

                    1    0   -1
                    2    0   -2
                    1    0   -1
# Results

Sobel Horizontal Edges

![Sobel_Horizontal_Edges](Sobel_Horizontal_Edges.png)

Sobel Vertical Edges

![Sobel_Vertical_Edges](Sobel_Vertical_Edges.png)

Sobel Edges Magnitude

![Sobel_Magnitude](Sobel_Magnitude.png)

# References
https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel
