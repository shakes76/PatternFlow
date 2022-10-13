Siamese Neural Network (SNN)
========
Classification model of the ADNI brain data [OASIS brain](https://adni.loni.usc.edu/) data set using [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf). 

**Author:** *Lukas Lobmaier (s4736267)*

* [SNN Architecture](#SNN-Architecture)<br>
* [Design Approach](#Design-Approach)<br>
* [Executing Code](#Executing-Code)<br>
* [Results](#Results)<br>
* [References](#References)

## ADNI brain data set
The ADNI brain data [OASIS brain](https://adni.loni.usc.edu/) datadata set contains mri brain scans of 1526 persons. For every person, 20 different slices from the 3D scan are provided. The two classed of the data set are AD (Alzheimer diseasse) and NC (Normal cognition).
The data set provides two folders for training and testing with two subfolders each corresponding to the two classes.  Each image file has an indicator for the specific person (XXXXXXX) and the specific slice number (YY). 

<p align="center">
    <img src="Picture/OverviewSNN.png" width="600" >
</p>
<p align="center">
    <em> Figure 1: ADNI brain data set  </em>
</p>

## SNN Architecture

The main idea of a Siamese Neural Network (SNN) , sometimes also refered as twin nural net [1], is to compare two inputs regarding their similarity. Therefore, the SNN has a unique structure compared to other neural nets.  </br>
The main strcuture constist of two identicall sub nets, which are processing each of the two input data samples. The outputs of these subnets can be refered as a complex feature mapping or fingerprint of the input sample, are then compared regarding similarity.

<p align="center">
    <img src="Picture/OverviewSNN.png" width="900" >
</p>
<p align="center">
    <em> Figure 1: Overview of SNN approach [Image Source](https://en.wikipedia.org/wiki/Animal)  </em>
</p>

The output of a classical SNN returns a single value:
* &nbsp;&nbsp;&nbsp;Same:    &emsp;Input samples are from the same class 
* Different:  &nbsp;Input samples are from different classes.

After the network has been tuned, the SNN provides good discriminative features. Due to this, the SNN can then be used to predict classes from new unseen data, but it can also be used on completly different, new classes, which have not been seen by the network during the training process.

One application are for SNNs lies in the field of face recognition. For example, a system tracking the acces right of people entering the building. Pure classification is not possible because the database onyl containts the faces of the recognized people who are allowed to enter the building. Especially due to the SNN learning how how to faces differ,  the database can kept quite small.

### Starting Point
The main layer stack of the first implementation is based on the  [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) implementation.

<p align="center">
    <img src="Picture/OverviewSNN.png" width="600" >
</p>
<p align="center">
    <em> Figure 2: Starting point for SNN layer stack [source](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)  </em>
</p>

This first implementation makes use of the implemented [BCEloss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) function of Pytorch.

### Custom Loss Function
For the next implementation, a custom loss function is introduced. The mathematical foundation  of this implementation is based on the paper [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)

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
To improve the accuracy, a classification net was inroduced. With three input vectors, the output gives back one of the two classes. 
=> This could increase the accuracy, but in this case it does not.
&nbsp;=> Probably due to the poor feature extraction of the SNN.
&nbsp;&nbsp;=> Might be worth looking into as soon as reaching 70% accuracy.

###  Training on same slice numbers
Due to the data corelation between the different slices, the datasets cannot be shuffeld randomly as before. The following procedure is used while loading the data:
1. Sorting data:  Sorting the data after person and slice number
2. Define sets: 20 slices are formed to one set/brain
3. Shuffle sets: Nessesary to be able to split up train and validation set
4. Shuffel slices:  Same slice numbers are shuffled between sets (not used in the final implementation)

The following hints are important if implementing a dataset for the ADNI dataset:
 + Pyhton function [glob()](https://github.com/python/cpython/blob/3.10/Lib/glob.py) returns image paths in random order
 + Inonsistens slice labeling in dataset
  + Patient set 808819 slice 1 labeled with 88 (808819_88.jpeg)
  + Patient set 218391 slice 1 labeled with 78 (218391_78.jpeg)

### Using ResNet
The next step was implementing a ResNet approach for the siamese branch. The residual and identiy blocks are build up from two convolutional layers with an skip connection. The convoltuional blocks use a convolutional layer with stride=2 to downsample the input image, while possible increasing the number of output channels. (=> [ResNet](https://arxiv.org/abs/1512.03385)). 

<p align="center">
    <img src="Picture/resnet.png" width="900" >
</p>
<p align="center">
    <em> Figure 3: ResNet  </em>
</p>

The used ResNet uses a image input size of 105x105, therefore the slice images have to be resized from the original size before feeding into the neural net. 
=> Even togh, the network is able to prevent same output for different inputs, the accuracy level stays at the same level.
### Using slices as channels

Maybe not all slices can be used for detecting the class. Therefore all 20 slices of a single person are packed into different channels of the tensor. This results in a sample size of 
20x240x256. 

### 3D Approach
To be able to use 3D convolutional layers, the slices have to loaded as third dimension and not as channels. This can easily be done by adding an additonal dummy dimension.
```Python
torch.unsqueeze(image3D, dim=0)
```
3D convultional layers are used to keep the information about the corelation of the different slices.

<p align="center">
    <img src="Picture/3Dslices.png" width="300" >
</p>
<p align="center">
    <em> Figure 4: 3D data trough stacking slices  </em>
</p>

### 3D Augmentation
Due to the corelation between the slices, the random augmentation from the previous implementation have to be modified. 

The used augmentation for randomly flipping the images vertically and horizontally have to be applied on all 20 slices at the same time. 

The same randomized crop and resize augmentation is applied to the two input samples of the two patients.

Furthermore a randomized blackout augmentation was introduced which is applied indivudaly on all 20 slices across the two input sampes.

<p align="center">
    <img src="Picture/augmentation.png" width="900" >
</p>
<p align="center">
    <em> Figure 5: 3D-Augmentation shown on single slice </em>
</p>

### Reducing the ammount of used slices
On sugestion of a tutor, the amount of used slices per person was reduced.
=> Improves training time 
=> Training loss is not consitnely decreasing but makes weird steps instead
=> Runs into the problem, same input =same output

-> Another Input from Tutor: You need all slices

###3D ResNet approach

The previously used ResNet approach was adapted to work with the 3D input data. The paper [Improving Deep Neural Network Interpretation for Neuroimaging Using Multivariate Modeling](https://link.springer.com/content/pdf/10.1007/s42979-022-01032-0.pdf) has acted as a guide for the 3D adaption of the implemented ResNet.

The final implementation of the 3D ResNet uses an input size of 210x210. The images only have to be cropped to the relevant area. For all brain scans the size of 210x210 is sufficent. Due to the preperation layer of the ResNet branch, the input does not have to be resized.

<p align="center">
    <img src="Picture/resnet3D.png" width="900" >
</p>
<p align="center">
    <em> Figure 1: ResNet 3D  </em>
</p>

For the convolutional residual blocks 3 and 4, a stride with (2,1,1) is used, to deacrease the number in the third dimensions but not in the other two.

Tuning the batch size, introducing a training step scheduler and gradient clipping, could increase the accuracy of the SNN. Nevertheless these helped to reduce the training time to archieve the "state of the art". 

Adding a deeper structure by using more residual blocks for the ResNet approach, didnt increase the accuracy. It only led to a drastical increase in training time and model size.

### One Shot image recognition
Until here, for the one shot image recognition only one sample person of two classed was used. Instead of using one reference image for both classes, the number could be increased to 10.

Averaging the slices itself across the patients, does not provide a improvement. This is mainly due to the orientation and location of the brain centers beeing different between patients.

A more promising approach is to look at the average of the output feature vector of the SNN.

### Outlook:
Further improvements for the SNN implementation would be the introduction of mining stargeies and a driplett loss. This provides the SNN with a positive (same class) and negative (differnt) class. This should increase the accuracy of the SNN drastically.
Furthermore, the previously used approach for a additional classification network could be used again. An aditional improvement could be made my tuning the SNN while training the classification Net.
Another approach would be to implement true 3D augmentation.

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
* https://github.com/python/cpython/blob/3.10/Lib/glob.py
* https://arxiv.org/abs/1512.03385