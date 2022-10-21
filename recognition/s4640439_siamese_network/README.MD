# Siamese Networks for Alzheimer's Disease Classification Using MRI Images

## Description and Problem
This project aims to use ADNI brain MRI images to classify Alzheimer's disease. Using raw images, the algorithm outputs a prediction of whether or not the pictured brain has Alzheimer's disease. This is done using a combination of a Siamese Neural Network and a Binary Classifier Neural Netowrk.

## How the Algorithm Works
Siamese Networks are essentially a slightly modified version of a CNN. How they work is, you pass two images samples into the model, one after the other. The model transforms these images into vector embeddings. Then, a distance is computed between these vectors, using one of potentially many distance metrics. Ideally, two images of the same class are very close in distance and images of different classes are very far in distance. The loss function and optimiser then work to update the weightings to move towards the ideal behaviour.

After the Siamese Network is trained, you then have a transformer which converts images to vector embeddings, keeping similar images close in distance together.

You then use these embeddings to train a dense layered Binary Classifier.

With luck, your Binary Classifier can then be used to accurately predict images by first converting test images to embeddings, and then classifying them into either a positive or negative class.

## Results
Unfortunately, as of the current version, this implementation has failed to construct a suitable classifier.

All attempts at training a Binary Classifier using my Siamese Model to generate embeddings led to the Classifier getting stuck at a 51% accuracy (the ratio of negative to positive samples).

I tried many different model structures, as well as tweaked various hyperparameters, but was not able to get the Siamese Model to generate satisfactory embeddings with which to classify with.

This leads to the Binary Classifier quickly getting stuck in a local minimum, unable to differentiate between the classes just by the embeddings.

Running principal component analysis revealed that the issue was with the siamese model. Taking the two principal components with the highest variance and plotting them for each embedding resulted in the following scatter plot:
![PCA graph for first two principal components of data embeddings](images/PCA.png)

As can be seen, jsut considering the two components with the highest variance, there is a major overlap between the two classes. Thus, it is no wonder that the binary classifier was unable to assess the data sufficiently.

I attempted many different tweaks to my Siamese Model in order to try to improve the embeddings.

Techniques attempted include:
* Trying various batch sizes in range (32, 128)
* Trying various epochs num in range (30, 100)
* Changing structure of model - size of convolutions, number of convolutions, strides, max padding
* Changing the margin in the loss function in the range (0.1, 0.5)

If more time were available, I would try to train the Siamese Model using various other loss functions instead, for example Triplet Loss. Additionally, I also would have liked to try different distance metrics for the embeddings.

## Running the Code
### Dependencies
Requires Python version 3.9 or above (for type hinting)
Requires tensorflow version 2.8.2 or above
Requires numpy version 1.21.3 or above
Requires matplotlib 3.4.3 or above
Requires pandas 1.3.2 or above
Requires scikit-learn 1.0.1 or above

### Dataset and Pre-processing
Original dataset sourced from: [ADNI dataset for Alzheimer's disease](http://adni.loni.usc.edu/)
Pre-processed dataset (used in this project) available from: [UQ Blackboard](https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI)

### Instructions
1. Run dataset.py - being sure to adjust path constants to match your personal setup
2. Run modules.py - adjusting image size and embedding shape as necessary
3. Run train.py - being sure to change the model save directory constant to where you would like to save your models
4. Run predict.py - being sure to change the path names for the test data
