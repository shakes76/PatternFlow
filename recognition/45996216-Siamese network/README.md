# Classifying the Alzheimer’s disease based on the  Siamese network



## Introduction to Siamese network:

​	Siamese network is a network that will receive two input images at a time. These two image will both go through the network. For a binary question, like identifying whether a brain has Alzheimer, if they are in the same class, the label would be true, would be false otherwise.

​	The below image is an example of Siamese network.

![1](C:\Users\xxx\Desktop\45996216-Siamese network\1.png)



## How Siamese network Works in General:

1. define a neural network, the outputs are an array. 

2. put two samples into this neural network and record they are in the same class or not

3. obtain the output of two images

4. calculate the Euclid distance between these images

5. put distance to output layer

   Make a prediction for a single image:

   We can choose two images we already known which the first one is brain which has Alzheimer, another one is not.

   We can create two samples like [image, known_image(label==True)],[image, known_image(label==Flase)]

   Based on these two result, we can make a prediction for a single image.

    

## How to use :

1. You can run the demo.ipynb file to run the whole program
2. Put train and test image file directly to the main folder. like this:![2](C:\Users\xxx\Desktop\45996216-Siamese network\2.png)

The train and test image can be downloaded on http://adni.loni.usc.edu/ or on the COMP3710 blackboard page for UQ students.

Run .py file as this process: dataset.py modules.py train.py predict.py.

## Dependencies Required:

numpy

tensorflow

keras  # independent version not keras embedded into tensorflow

## Results:

This model finally got 96.6% accuracy on the test set.

99.9% accuracy on the training set.

## Reference:

G. Koch, R. Zemel, R. Salakhutdinov et al., “Siamese neural networks for one-shot image recognition,” in ICML deep learning workshop, vol. 2. Lille, 2015, p. 0.



