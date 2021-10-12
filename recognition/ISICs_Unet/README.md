# Segmenting ISICs with U-Net

COMP3710 Report recognition problem 3 (Segmenting ISICs data set with U-Net) solved in TensorFlow

Created by Christopher Bailey (45576430)

## The problem and algorithm
The problem solved by this program is binary segmentation of the ISICs skin lesion data set. Segmentation is a way to label pixels in an image according to some grouping, in this case lesion or non-lesion. This translates images of skin to masks representing areas of concern for skin lesions.

U-Net is a form of autoencoder where the downsampling path is expected to learn the features of the image and the upsampling path learns how to recreate the masks. Long skip connections between downpooling and upsampling layers are utilised to overcome the bottleneck in traditional autoencoders allowing feature representations to be recreated.

## How it works
A four layer padded U-Net is used, preserving skin features and mask resolution. The implementation utilises Adam as the optimizer and implements Dice distance as the loss function as this appeared to give quicker convergence than other methods (eg. binary cross-entropy).

The utilised metric is a Dice coefficient implementation. My initial implementation appeared faulty and was replaced with a 3rd party implementation which appears correct. 3 epochs was observed to be generally sufficient to observe Dice coefficients of 0.8+ on test datasets but occasional non-convergence was observed and could be curbed by increasing the number of epochs. Visualisation of predictions is also implemented and shows reasonable correspondence. Orange bandaids represent an interesting challenge for the implementation as presented.

### Training, validation and testing split
Training, validation and testing uses a respective 60:20:20 split, a commonly assumed starting point suggested by course staff. U-Net in particular was developed to work "with very few training images" (Ronneberger et al, 2015) The input data for this problem consists of 2594 images and masks. This split appears to provide satisfactory results.

## Using the model
### Dependencies required
* Python3 (tested with 3.8)
* TensorFlow 2.x (tested with 2.3)
* glob (used to load filenames)
* matplotlib (used for visualisations, tested with 3.3)

### Parameter tuning
The model was developed on a GTX 1660 TI (6GB VRAM) and certain values (notably batch size and image resolution) were set lower than might otherwise be ideal on more capable hardware. This is commented in the relevant code.

### Running the model
The model is executed via the main.py script.

### Example output
Given a batch size of 1 and 3 epochs the following output was observed on a single run:
Era | Loss | Dice coefficient
--- | ---- | ----------------
Epoch 1 | 0.7433 | 0.2567
Epoch 2 | 0.3197 | 0.6803
Epoch 3 | 0.2657 | 0.7343
Testing | 0.1820 | 0.8180


### Figure 1 - example visualisation plot
Skin images in left column, true mask middle, predicted mask right column
![Visualisation of predictions](visual.png)

## References
Segments of code in this assignment were used from or based on the following sources:
1. COMP3710-demo-code.ipynb from Guest Lecture
1. https://www.tensorflow.org/tutorials/load_data/images
1. https://www.tensorflow.org/guide/gpu
1. Karan Jakhar (2019) https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c
