# A Siamese Network for Classifying Alzheimer's on the ADNI Dataset

This project aims to implement a classifier based on a Siamese Network architecture to classify Alzheimer's disease in the ADNI dataset.

## Dataset

The ADNI data consisted of a test, training and validaiton set. The training set has 21,520 images in total, with 11,120 of the brain scans beloging to patients without Alzheimer's and 10,400 belonging to patients with Alzheimer's.

## The Classifier based on a Siamese Network

As a brief introduction, a Siamese network is a pair of Convolutional Neural Networks that share the same weights, and each producing a vector. Each network in the Siamese network is fed an image simultaneously, and from this image, each network will produce a single vector. With these vectors, which are sometimes also referred to as embeddings, a distance metric can be computed between them (such as L1 distance in my case), which can be used to determine the similarity between the images according to the Siamese network. Finally, the distance between the embeddings can be fed into a final sigmoidal function, which classifies whether the pairs of images belong to the same or different classes.

## Dependencies

This project has the following dependencies:

* tensorflow==2.9.1
* Pillow==9.2.0
* numpy==1.23.3
* scikit-learn==1.1.2
* os

## Results




## References
