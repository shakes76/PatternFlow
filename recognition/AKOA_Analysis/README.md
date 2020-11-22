# PROJECT
OAI AKOA knee lateral classification
## PROJECT OVERVIEW
The lateral classification of the OAI AKOA knee dataset(left or right sided knee) with a minimum accuracy of 90%.
## DEPENDENCIES
* Python 3.7
* TensorFlow 2.3
* Matlablib
  * pyplot
* Numpy
* sklearn.model_selection
  * train_test_split
* PIL
  * Image
## ALGORITHM
The model is a convolutional neural network that consists of one 2d convolutional layer and dense layers on the top to perform the classification. A convolutional neural network(CNN) is a deep learning algorithm that takes in an input shape and is able to differentiate it from another image. The purpose of the convolutional layer is to help classify the image by breaking the image into features giving the network more of an understanding of the images. As the images only had to be classified into two categories, the need for more layers was deemed unnecessary.
## DATA SPLIT
The dataset had over 18,000 images, each labeled left or right. The training data was 60% of the dataset, the testing data was 25% of the dataset and the validation data was 15% of the dataset. For more information on the dataset, refer to https://nda.nih.gov/oai/ for more information.
## EXAMPLE OUTPUTS
### Training and Validation Accuracy
![Accuracy graph](https://raw.githubusercontent.com/josh-lim1234/PatternFlow/topic-recognition/recognition/AKOA_Analysis/graphs/accuracy.png)
### Last Epoch Statistics
![Last Epoch Statistics](https://raw.githubusercontent.com/josh-lim1234/PatternFlow/topic-recognition/recognition/AKOA_Analysis/graphs/lastepochstats.png)
