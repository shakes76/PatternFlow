# Improved UNET on ISICs Dataset

This uses the Improved UNET on the ISICs dataset to segment the images into background and skin cancer. 

# Problem 

The ISICs dataset starts with a normal image of the skin cancer, as well as a segmentation image paired with it. Our model must take these images to train using the Improved UNET Model and then be able to segment it on its own using just the skin cancer images. It then compares this to expected output to calculate the efficiency. This output should be above 80%. The below images show what the input image is, with the expected output alongside it.

![skin image]("Images/Input Image.jpg")
![segment image]("Images/Expected Output.png")

## Improved UNET Model

![UNET Model image]("Images/Improved UNET.png")
This is the image of the Improved UNET Model, as given by the paper *"Brain Tumor Segmentation and Radiomics Survival Prediction"*. The model works similar to the standard UNET model. It has two halves, encoding and decoding. It has the same U-shape as the standard UNET, but adds extra concatenation and adding of layers for a more efficient model. The improved UNET also uses Leaky ReLU activation.

Each layer of the Improved UNET takes a 3x3x3 convolution and adds it to the context module of the same layer. It does this multiple times, from a starting size of 16 until it reaches size 256.  This is where upsampling begins and is the second half of the model. This takes the output that is saved from the first half of the model, and performs upsampling and localisation. In the last 3 layers, it does a segmentation layer with a softmax for the last. My model also adds a sigmoid activation to restrict the output to be within 0 and 1.

## Dependencies 
1.  Python 3.9.7
2. Tensorflow 2.6.0
3. Matplotlib 3.4.3

## Usage 

### Using model.py
This is simply the model support file. It should not be executed on its own. However **build_model()** is the function called to return a model of the Improved UNET.

### Using driver.py
The driver file works by making a copy of the Improved UNET model, then loads the data. It loads the images into a dataset which then gets converted into arrays that the model can use for fitting. The model is then compiled using the adam optimisier, and dice coefficient for loss and its metrics before fitting begins.
Then the output is given to be plotted for all images, loss and accuracy calculation.

The number of epochs can be modified using the *epoch* variable. For this run, 100 epochs were run as this seemed to provide sufficient results and learning.

## Output
![model output]("Images/Sample Output.png")

Here is the output comparing the original, expected and actual output. There are some bugs within this implementation regarding the dice coefficient value being greater than 1. This could be the reason for the less-than optimal output from the model. However we can still see that the output from **model.predict()** is close to the expected output in the right row. When looking at the 'accuracy' metric given to **model.compile**, we can see that this value reaches around 60% accuracy. This is above the 80% expected, however with more epochs or a bigger batch size, a higher accuracy could be achieved. The batch size of 16 was selected as this was the memory limit of my RTX3070, higher sizes would exceed the memory capacity of my GPU.

## Training and Test split
I chose to use a 90/10 split for training and testing. This is so that model has more images to refine its learning, reading for when a test set is given for its efficiency testing.


#### References
[Paper on Improved UNET Model](https://arxiv.org/pdf/1802.10508v1.pdf)

[Dice Coefficient Function](https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c)

[ISICs Dataset from 2018 Challenge](https://challenge2018.isic-archive.com/)