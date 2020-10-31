# Brain MRI (oasis dataset) Semantic Image Segmentation using improved u-net

## algorithm description

On high level u-net has contracting path (on the left side) and an expansive path. The contracting path of the improved u-net is built using context modules which are pre-activated residual blocks, each consists of batch normalization layer, activation, 3x3 conv, BN, Activation, 3x3 conv and addition layer. The down sampling is done using 3x3 convolution with 2x2 stride. The expansive path is built using localization module followed by up sampling modules. The localization module has 3x3 convolution layer followed by 1x1 convolution that half the number of feature maps. The up sample module is made of 2x2 up sample layer that expand the image size followed by 3x3 convolution that doubles the number of feature maps. Improved u-net uses the traditional skip connections but also has additional residual connection aggregating segmentation output from different levels of the architecture to augment the final output.  
  
## semantic image segmentation problem
This is about identifying the four segments or parts in the brain from MRI images. It is basically pixel wise classification of the image i.e. identify for each pixel to which class it belongs.

## train parameters and procedure
The used activation is leaky Relu with alpha = .01.
The used optimizer is Adam and I used learning rate of .0005 
The paper used dice similarity loss to train their model, but I couldn’t implement it, so I used Categorical cross entropy
I used multiples of 16 filters at each level of the network exactly as specified by the paper.  
I trained the model for 200 ephocs with early stopping call back to restore the highest validation accuracy weights and stop training after 30 ephocs of no improvement in validation accuracy. Training stopped after 35 ephocs

The dataset was already split into training, validation and test data sets. Validation dataset is useful during training to monitor training for overfitting and I used test dataset to assess model generalization capability on a set not seen during training. 
## dependencies and data pre-processing 
Training data was normalized by subtracting mean and dividing by standard deviation and then normalizing the pixel values between 0-1. I noticed that normalizing the data this way results in a more stable training vs dividing by 255.
The label images as well need to be pre-processed and converted to one hot encoding representation. 





## output 
The below results show the prediction of the four segments vs ground truth at val_accuracy of .87 and dice coeff .62
   ![image]( C:\Users\karee\PatternFlow\recognition\MySolution\res block.JBG)

## model summery


