Siamese Neural Network (SNN)
========
Classification model of the ADNI brain data[OASIS brain](https://adni.loni.usc.edu/) data set using [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

**Author:** *Lukas Lobmaier (s4736267)*

* [SNN Architecture](#SNN-Architecture)<br>
* [Design Approach](#Design-Approach)<br>
* [Executing Code](#Executing-Code)<br>
* [Results](#Results)<br>
* [References](#References)

## SNN Architecture

The main idea of a Siamese Neural Network (SNN) , sometimes also refered as twin nural net [1], is to compare two inputs regarding their similarity. Therefore, the SNN has a unique structure compared to other neural nets.  </br>
The main strcuture constist of two identicall sub nets, which are processing each of the two input data samples. The outputs of these subnets can be refered as a complex feature mapping or fingerprint of the input sample, are then compared regarding similarity. 

<p align="center">
    <img src="Picture/OverviewSNN.png" width="600" >
</p>
<p align="center">
    <em> Figure 1: Overview of SNN approach  </em>
</p>
https://en.wikipedia.org/wiki/Animal

The output of a classical SNN returns a single value:
* Same:    &nbsp;&nbsp;&nbsp;&emsp;Input samples are from the same class 
* Different:  &nbsp;Input samples are from different classes.

After the network has been tuned, the SNN provides good discriminative features. Due to this, the SNN can then be used to predict classes from new unseen data, but it can also be used on completly different, new classes, which have not been seen by the network during the training process.

One application are for SNNs lies in the field of face recognition. For example, a system tracking the acces right of people entering the building. Pure classification is not possible because the database onyl containts the faces of the recognized people who are allowed to enter the building. Especially due to the SNN learning how how to faces differ,  the database can kept quite small.

## Design Approach
### Implementing Net
### Starting Point
The main layer stack of the first implementation is based on the  [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) implementation.

<p align="center">
    <img src="Picture/OverviewSNN.png" width="600" >
</p>
<p align="center">
    <em> Figure 1: Starting point for SNN layer stack  </em>
</p>
https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

This first implementation makes use of the implemented [BCEloss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) function of Pytorch.

### Custom Loss Function
For the next implementation, a custom loss function is introduced which is based on the work [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)


```Python
class ContrastiveLoss(torch.nn.Module):

	def __init__(self, margin=2.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, output1, output2, label):
		euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

		loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) 			+ (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

		return loss_contrastive

```
To use this, the SNN outputs the feature vectors instead of the calculated L1 norm.  With this custom loss function the training time and impact could be increased.

###Classification Net
To improve the accuracy, a classification net was inroduced. With three input vectors, the output gives back one of the two classed. 
=> This hould increase the accuracy, but in this case it does not.
&nbsp;=> Probably due to the poor feature extraction of the SNN.
&nbsp;&nbsp;=> Might be worth looking into as soon as reaching 70% accuracy.

### Training on same slice numbers
Due to the data corelation between the different slices, the training and testing set.
Sorting dataset
Shuffeling sorted data per person
--Different sublabels for slices
--Gob is ranndomly reading image paths

Shuffeling same slices across different persons
### Using ResNet
Using ResNet approach for the siamese branch
=> Same result


### Using slices as channels

Maybe not all slices can be used for detecting the class.
Packing all slices of a person in  different channels => output 20x240x256

Augmenting data per person - Random pytorch function is not working anymore
-RandomCrop
-Random Blackout
-Random HorizontalFlip
-Random Vertical Flip

### 3D Approach

3D Convolution - Better understanding connection between the slices
Transfering output to three dimensional with one channel 1x20x240x256

3D Convuoltional approach of the orignal papper

### Reducing the ammount of used slices
=> Training with ResNet
improves to actually train
runs quite fast
=> accuracy makes weird steps, converges but does not reach the "state of the art" of 53%
=> does run into the problem, same input =same output


-> Input from Tutor: You need all slices
### Improving real 3D Net

Adding deeper structure
More conv layers -> Might add resnet approach 
=> Training time drastically increases


Adding ResNet approach to 3D data
last idea to improve


### outlook:
-Mining strategies
-Driplett Loss  - positive and negative sample – 
-Classifier training SNN s well

Vgg 11
Lin 128 resnet vgg  embedded ize


https://arxiv.org/pdf/2011.00840.pdf
https://arxiv.org/pdf/1909.01098.pdf

## Executing Code

train.py
train_3D.py


## Results

Problems: SNN tends to output the same for all input => Too complex task for the NET =>

SNN tends to randomize guessing to solve towards the 55% accuracy







## References
* https://adni.loni.usc.edu/
* https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
* https://en.wikipedia.org/wiki/Siamese_neural_network
* https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
* http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf