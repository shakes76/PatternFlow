# Assignment Report for COMP3710 by Bo Peng (s4652534)
Create a classifier based on Siamese network to classify Alzheimer’s disease (normal and AD)

# Dependency
- python(3.9.5)
- pytorch(1.7.1+cu110)
- torchvision(0.8.2+cu110)
- numpy(1.21.2)
- matplotlib(3.4.3)
- PIL (1.1.7)

# Dataset
ADNI brain data set will be automatically loaded during training and testing by setting the variable "dataset_dir" (in both train.py and predict.py) as the dictionary where the ADNI brain data set is stored. 

ADNI brain data set is split into two sets, i.e., a training set and a testing set, where the former includes 10400 AD + 11120 NC samples and the latter includes 4460 AD + 4540 samples.

# Implementation Details
In this project, the Resnet-34 [] is used as the backbone, followed by a fully-connected layer, a batch normalization layer and a classification layer. In addition to the widely used Cross-Entropy classification loss, the hard-batch sofr-margin triplet loss [] is utilized based on Siamese network for learning more represenative intermediate represenations (the output of the backbone) to facilitate classification. The weight of the two losses are equal.



# Training
To start training, simply run
> python train.py

after training 500 epoches, one can obtain a classifier with an accuracy of 0.8 on the test set. The trained weights of the classifier is saved into the file best_model.pkl in the current dictionary.

# Test
To start testing, simply run
> python predict.py

This results intesting the classification perforamnce of the trained model by demonstraing predictions for 15 randomly selected testing samples in Demo.jpg.





