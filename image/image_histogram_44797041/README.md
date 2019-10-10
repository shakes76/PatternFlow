# Tensorflow implementation of gaining a histogram of an image

The histogram produced is a representation of the tonal distribution of the flattened image.
For each tonal value, the count of the number of pixels for this tonal value is computed. By
viewing the histogram, the entire tonal distribution is given at a glance. Note that for colour images, each channel should be independently input to gain a histogram for each colour channel. 
