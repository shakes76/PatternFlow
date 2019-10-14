# Histogram Equalisation
This directory contains a Tensorflow implementation of the histogram equalisation algorithm.

## Description

Histogram equalisation is an image processing technique used to enhance contrast in images.
It is most commonly used when image data is represented by close contrast values.

It works by first calculating the histogram of a given image, which is essentially a distribution
representing tonal intensity. 

## Implementation
equalize_hist(image, nbins=256, mask=None)

    Returns an image after histogram equalisation

    Parameters
    -----------
    image : array
        Image to be equalised
    nbins : int optional
        Number of bins for the histogram
    mask: array optional
        Array of bools (as 1s & 0s) which restricts the areas used
        to calculate the histogram

    Returns:
    -----------
    output : array
        Float32 array representing the equalised image

## Getting Started

#### Dependencies
* Python 3.6
* Tensorflow 1.14
* Tensorflow-probability 0.7

#### Installation
* git clone https://github.com/drussell13/PatternFlow.git

#### Simple Example
    from skimage import data
    import matplotlib.pyplot as plt
    from equalize_hist import equalize_hist
    
    img = data.moon()
    img_eq = equalize_hist(img)
    
    plt.imshow(img_eq)
    plt.show()
    

## Examples

#### Colour Images
    from skimage import data
    import matplotlib.pyplot as plt
    from equalize_hist import equalize_hist
    
    img = data.astronaut()
    img_eq = equalize_hist(img)
    
    plt.imshow(img_eq)
    plt.show()
    
#### Applying a Mask
    from skimage import data
    import matplotlib.pyplot as plt
    from equalize_hist import equalize_hist
    
    img = data.moon()
    img_eq = equalize_hist(img)
    
    plt.imshow(img_eq)
    plt.show()