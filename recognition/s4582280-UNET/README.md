# Credits
This was created by Dylan Baynes (student number, 45822801) for CSSE3710

# Task
CSSE3710 - Semester 2 2022
1. Segment the ISIC data set with the Improved UNet with all labels having a minimum Dice similarity
coefficient of 0.8 on the test set. [Easy Difficulty]

# The Algorithm
## Description
The algorithm which was created to solve the task of segmenting the ISIC data set, this process is essentially seperating 
different features of the dataset and creating a sort of "computer vision" where the UNET seperates different features
as layers and outputs a new mask which will show what the network has identified. This is a highly relevant problem in 
todays society as researchers and engineers are always looking for ways to improve AI technology as well as to make our
jobs easier by having networks identify features less experienced people or even professionals may look past.

## How it works
The algorithm works in a simple way, by labelling each pixel in the image a unqiue colour as to what the network thinks
is being represented after it performs feature extraction. The following is an example of how this works:
![towardsdatascience, 2022, image segmentation](https://miro.medium.com/max/720/1*nXlx7s4wQhVgVId8qkkMMA.png)

The above image was provided by towardsdatascience.com, see image alt text for full source.

Compared to other segmentation methods this is able to produce high quality images as it's simply "overlaying" it's own
vision onto a premade image, meaning it must match the resolution

## Running the algorithm
It is very easy to run the algorithm, simply run the train.py file and it will begin the training process while producing
a nice set of accuracy and loss graphs to show how well it performed overtime.

### Epochs
The base number of epochs is set to 4 so that anyone without a proper GPU can run this within a decent timeframe, however
this part of the code should be directly increased to 8+ to see a dice similarity score of above 0.8.

## Preprocessing
### Data Files
The data which is used for this project has had a small amount of adjustment made, primarily each folder must only contain
the relevant data images and not the superpixel files some of these contain.

### Code-based
Once the data is loaded, the training data is randomized as well to improve the variance of the testing. We also split
the data into training, testing and validation sets, with a 50-25-25 (%) split respectively. This is done to provide 
enough training data for the process but also enough so that it can properly validate and test itself.



## Dependencies
- Python 3
- Tensorflow
- Keras
- ISIC dataset files (placed into "Data" folder), you need both the true masks and the normal testing files


1. The readme file should contain a title, a description of the algorithm and the problem that it solves
(approximately a paragraph), how it works in a paragraph and a figure/visualisation.
2. It should also list any dependencies required, including versions and address reproduciblility of results,
if applicable.
3. provide example inputs, outputs and plots of your algorithm
4. The read me file should be properly formatted using GitHub markdown
5. Describe any specific pre-processing you have used with references if any. Justify your training, validation
and testing splits of the data.
