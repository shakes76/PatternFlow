# Classifying Knee Laterality with a CNN
This is the recognition project component of COMP3710.

s4361465 Kelsey McGahan 

Given a database of MRI images of left and right knees I was able 
to build a Convolutional Neural Network (CNN) that performed binary
classification. The data was supplied and preprocessed by the course
and was sourced from the 
[Osteoarthritis Initiative](https://nda.nih.gov/oai/). 

The images were labelled with their sidedness in the filename. 
The dataset contained 18680 images with an approx. 40/60 split
of left/right images. The images were (228, 260) and RGB. The 
pixel values are between 0 and 1. 
Before training the model I re-sized the images to (64,64) to 
decrease training time. 
Below are example of left and right knees.

![Left Right Knee Image Example](readmeImages\left_right_example.png)

Binary CNN classification of left and right sidedness of images of knees.

- [ ] Description of algorithms

- [ ] Problem that algorithm slves

- [ ] how it works ( in a paragraph and a figure/visualisation)

- [ ] list any dependencies (does this include libraries??)

- [ ] provide example outputs and plots of algorithm code

- [ ] describe and justify your training, validation adn testing split of the data


## Dependencies
* Jupyter Notebook
* Tensorflow (2.1.0)
* Python (3.7.9)
* matplotlib (3.3.1)
* numpy (1.19.1)
* cv2 (4.4.0)
* wget (3.2)
* sklearn (0.23.2)

## Files in Repository
* ``Driver-Script.ipynb`` - Interactive notebook to import, 
unzip, pre-process the data, then call the model and plot results.
* ``Classify_Knee_Laterality_Model.py`` - Tensorflow model.
* ``README.md`` -  This file.
* ``readmeImages/*`` - Images for this README.