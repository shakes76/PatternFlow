# Improved UNET on ISICs Dataset

This uses the Improved UNET on the ISICs dataset to segment the images into background and skin cancer. 

# Problem 

The ISICs dataset starts with a normal image of the skin cancer, as well as a segmentation image paired with it. Our model must take these images to train using the Improved UNET Model and then be able to segment it on its own using just the skin cancer images. It then compares this to expected output to calculate the efficiency. This output should be above 80%. The below images show what the input image is, with the expected output below it.

![skin image](https://github.com/AndrewLuong6/PatternFlow/blob/topic-recognition/recognition/45820188-UNET/Images/Input%20Image.jpg?raw=true)

![segment image](https://github.com/AndrewLuong6/PatternFlow/blob/topic-recognition/recognition/45820188-UNET/Images/Expected%20Output.png?raw=true)

## Improved UNET Model

![UNET Model image](https://github.com/AndrewLuong6/PatternFlow/blob/topic-recognition/recognition/45820188-UNET/Images/Improved%20UNET.png?raw=true)
This is the image of the Improved UNET Model, as given by the paper *"Brain Tumor Segmentation and Radiomics Survival Prediction"*. The model works similar to the standard UNET model. It has two halves, encoding and decoding. It has the same U-shape as the standard UNET, but adds extra concatenation and adding of layers for a more efficient model. The improved UNET also uses Leaky ReLU activation.

Each layer of the Improved UNET takes a 3x3x3 convolution and adds it to the context module of the same layer. It does this multiple times, from a starting size of 16 until it reaches size 256.  This is where upsampling begins and is the second half of the model. This takes the output that is saved from the first half of the model, and performs upsampling and localisation. In the last 3 layers, it does a segmentation layer with a softmax for the last. My model also adds a sigmoid activation to restrict the output to be within 0 and 1.

## Dependencies 
1. Python 3.9.7
2. Tensorflow 2.6.0
3. Matplotlib 3.4.3

## Usage 

### Using model.py
This is simply the model support file. It should not be executed on its own. However **build_model()** is the function called to return a model of the Improved UNET.

### Using driver.py
The driver file works by making a copy of the Improved UNET model, then loads the data. It loads the images into a dataset which then gets converted into arrays that the model can use for fitting. The model is then compiled using the adam optimisier, and dice coefficient for loss and its metrics before fitting begins.
Then the output is given to be plotted for all images, loss and accuracy calculation.

The number of epochs can be modified using the *epoch* variable. For this run, 1000 epochs were run as this seemed to provide sufficient results and learning.

### Hyper Parameters:
**Batch Size**: 16

**Epochs**: 1000

**Image Size**: (96 x 128)

## Output
![model output](https://github.com/AndrewLuong6/PatternFlow/blob/topic-recognition/recognition/45820188-UNET/Images/Sample%20Output.png?raw=true)

Here is the output comparing the original, expected and actual output. After leaving it to run overnight with 1000 epochs, the dice coefficient value of 86.84% was found. The following graphs show the change of dice coefficient as well as the loss over the 1000 epochs that were run. It looks to peak above 90%, which is unexpected, but could be due to the large number of epochs or an error causing overfitting and leakage.

### Dice Coefficient 
![dice coefficient](https://github.com/AndrewLuong6/PatternFlow/blob/topic-recognition/recognition/45820188-UNET/Images/Dice%20Value.png?raw=true)

### Loss Value
![loss value](https://github.com/AndrewLuong6/PatternFlow/blob/topic-recognition/recognition/45820188-UNET/Images/Loss%20Value.png?raw=true)

## Training and Test split
I chose to use a 80/20 split for training and testing. This is so that model has more images to refine its learning, reading for when a test set is given for its efficiency testing.


#### References
[Paper on Improved UNET Model](https://arxiv.org/pdf/1802.10508v1.pdf)

[Dice Coefficient Function](https://www.jeremyjordan.me/semantic-segmentation/)

[ISICs Dataset from 2018 Challenge](https://challenge2018.isic-archive.com/)