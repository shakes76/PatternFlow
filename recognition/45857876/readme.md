# Knee MRI Image stylegan
student name: Mengyao Ma

## Algorithm description
**GAN** is short for **Generative Adversarial Network** proposed by Ian Goodfellow in 2014. The main structure of GAN includes a **Generator(G)** and a **Discriminator (D)**. 

## Difference between 

## Style mixing
This is about identifying the four segments or parts in the brain from MRI images. It is basically pixel wise classification of the image i.e. identify for each pixel to which class it belongs.

## Train parameters and procedure
The used activation is leaky Relu with alpha = .01.
The used optimizer is Adam and I used learning rate of .0005 
The paper used dice similarity loss to train their model, but I couldn’t implement it, so I used Categorical cross entropy, and used Dice Similarity Coefficient as a metric to monitor training. 
I used multiples of 16 filters at each level of the network exactly as specified by the paper.  
I trained the model for 200 ephocs.

The dataset was already split into training, validation and test data sets. Validation dataset is useful during training to monitor training for overfitting and I used test dataset to assess model generalization capability on a set not seen during training. 
## Dependencies and data pre-processing 
The test script download , unzip the dataset images. The methods that load and process dataset take the directory path were the images were downloaded as a parameter. An update to these file paths might be required for the algorithm to run. 

Training data was normalized by subtracting mean and dividing by standard deviation and then normalizing the pixel values between 0-1. I noticed that normalizing the data this way results in a more stable training vs dividing by 255.
The label images as well need to be pre-processed and converted to one hot encoding representation. 



## Output 
The below results show the prediction of the four segments vs ground truth of image 3 and 5 in test set. Training loss was .14 , validation loss .425 and DCS of test set was .8972.

![all3](C:\Users\karee\PatternFlow\recognition\all3.JPG)![c2](C:\Users\karee\PatternFlow\recognition\c2.JPG)![all5](C:\Users\karee\PatternFlow\recognition\all5.JPG)![c2_5](C:\Users\karee\PatternFlow\recognition\c2_5.JPG)![c3](C:\Users\karee\PatternFlow\recognition\c3.JPG)![c1_5](C:\Users\karee\PatternFlow\recognition\c1_5.JPG)![c3](C:\Users\karee\PatternFlow\recognition\c3.JPG)![c1](C:\Users\karee\PatternFlow\recognition\c1.JPG)


## model summery
