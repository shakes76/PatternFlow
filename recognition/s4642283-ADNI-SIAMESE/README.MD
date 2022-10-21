# A Siamese Network for Classifying Alzheimer's on the ADNI Dataset

Name: Matthew Shearer

Student Number: 46422833

This project aims to implement a classifier based on a Siamese Network architecture to classify Alzheimer's disease in the ADNI dataset.

## Dataset

The ADNI data that has been preprocessed by the COMP3710 was used, however, the data was pooled together and training, testing and validation splits were made using 60/20/20 proportions. The `dataset.py` file expects the ADNI data to be given in a folder in the same directory as the file in the following structure:

- ADNI_AD_NC_2D
  - AD_NC
    - test
      - AD
        - *.png
      - NC
        - *.png
    - train
      - AD
        - *.png
      - NC
        - *.png

## General Introduction to Siamese Architectures

Siamese Neural Networks (SNN) work by having a pair of Convolutional Neural Networks (CNNs) that share the same weights, with a pair of images being fed into each of the CNNs (i.e, there are two CNNs, each receiving an image that makes up a pair). Each CNN, which is sometimes referred to as the embedding model or embedding layer, then produces a vector and a distance computation is made on each of the CNNs vectors. The result of this computation is fed into a final sigmoid layer, which classifies whether the pair of images is similar or not. A general layout of an SNN can be seen below:

![Screenshot](images/snn_example.png)

## The Classifier based on a Siamese Network (the algorithm and the problem it solves)

As already explained, an SNN is a pair of CNNs with shared weights, with each CNN creating an embedding given some image. Extending this a little further, we can use the trained Siamese network's embedding CNN to classify individual images, since the CNN is essentially being trained to create an embedding for each class that will be dissimilar to another class, and so in a way it can be used for classification. Clearly, the problem this implementation is trying to solve is the detection of Alzheimer's in brain scans.

## Usage

The `train.py` file does not require any commandline arguments and can be called as long as the dataset is in the correct directory structure (which is necessary for the `dataset.py` file). The only requirement for `predict.py` is that a model name 'Classifier_Model' is given in the same directory that the file is in. Both `dataset.py` and `modules.py` do not need to be called in order for `train.py` to work. Following from the task sheet, `train.py` will train a model based on the architecture in `modules.py`, and will produce training and validation accuracy and loss plots, with a testing accuracy. A classifier will then be created using the embedded network in the SNN to classify images. Both the SNN and this classifier will be saved (with the classifier being the more important model).

The `predict.py` will take a file path from the directory `predict.py` is in as an argument in the command line (which expects 'AD' and 'NC' child directories similar to the data structure shown above) and then use the classifier model in the same directory to make predictions on the data, with the final accuracy being printed out.

## Results

Although several attempts and different architectures were created, the most successful one achieved a test accuracy of 85.35% for the SNN while the classifier achieved an accuracy over the entire dataset of 60.75%. The architecture that achieved this is given in the `modules.py` file. The training and validation accuracies and losses can be seen below (note that validation was named 'test' on accident, sorry for the confusion).

![Screenshot](images/acc.png)
![Screenshot](images/loss.png)

Evidently, although it trained for 50 epochs, the architecture seems to reach around a 80% accuracy on the validation within 25 epochs, and doesn't increase much higher than that. We can also see that the validation loss doesn't really decrease after 22 epochs, and the model is clearly overfitting after ~25 epochs. Using the embedding layer from this architecture, the classification model achieved a 60.75% accuracy on the entire dataset. I have tried a lot to improve this, but I just can't seem to get it going better, although I think the most obvious route to improve it would be to battle overfitting better. A screenshot of the classifier's accuracy output can be seen below:

![Screenshot](images/classifier_acc.png)

In addition to this, I've provided an example of the training output (note that there is not really an example input):

![Screenshot](images/example_training.png)

Hence, although the SNN does well at comparing a pair of brain scans and determining whether they both of Alzheimer's, both are Alzheimer-free or they are not similar, the classifier based on the SNN does not seem to perform as well.

## Dependencies

This project has the following dependencies:
* Python 3.9.0
* tensorflow 2.9.1
* Pillow 9.2.0
* numpy 1.23.3
* scikit-learn 1.1.2
* matplotlib 3.5.2
* os
* random

## Note on Hardware

I should note that the code (and the results above) come from me executing this code in a google colab environment.

## Note on Reproducability

It should be noted tht reproducing the results above might be a little difficult, since the data is randomly split, and so you each run might take less (or more) epochs to reach the ~80% validation accuracy. However, from repeated runs, I've found that it will almost always reach ~80% accuracy with 30-45 epochs.

## References
[1] https://keras.io/examples/vision/siamese_contrastive/ 

[2] https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
