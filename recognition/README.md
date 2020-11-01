# Brain MRI Semantic Image Segmentation using improved u-net

## Algorithm description

![arch](C:\Users\karee\PatternFlow\recognition\arch.JPG)

On high level u-net has contracting path (on the left side) and an expansive path. The contracting path of the improved u-net is built using context modules which are pre-activated residual blocks, each consists of batch normalization layer, activation, 3x3 conv, BN, Activation, 3x3 conv and addition layer. The down sampling is done using 3x3 convolution with 2x2 stride. The expansive path is built using localization module followed by up sampling modules. The localization module has 3x3 convolution layer followed by 1x1 convolution that half the number of feature maps. The up sample module is made of 2x2 up sample layer that expand the image size followed by 3x3 convolution that doubles the number of feature maps. Improved u-net uses the traditional skip connections but also has additional residual connection aggregating segmentation output from different levels of the architecture to augment the final output.  



![res_block](C:\Users\karee\PatternFlow\recognition\res_block.JPG)

## Semantic image segmentation problem
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


