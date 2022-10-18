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
The ADNI brain data [OASIS brain](https://adni.loni.usc.edu/) data
set contains MRI brain scans of 1526 persons. For every person, 20 different slices from the 3D scan are provided. The two classes of the data set are AD (Alzheimer's disease) and NC (Normal cognition).
The data set provides two folders for training and testing with two subfolders each corresponding to the two classes.  Each image file has an indicator for the specific person (XXXXXXX) and the specific slice number (YY). 

<p align="center">
    <img src=Picture/dataset.PNG width="600" >
</p>
<p align="center">
    <em> Figure 1: ADNI brain data set  </em>
</p>

## SNN Architecture

The main idea of a Siamese Neural Network (SNN), sometimes also referred to as a twin neural net [1], is to compare two inputs regarding their similarity. Therefore, the SNN has a unique structure compared to other neural nets. The main structure consists of two identical subnets, which are processing each of the two input data samples. The outputs of these subnets can be refered to as a complex feature mapping or fingerprint of the input sample, and are then compared regarding similarity.
<p align="center">
    <img src=Picture/OverviewSNN.PNG width="500" >
</p>
<p align="center">
    <em> Figure 1: Overview of SNN approach [Image Source Animals](https://en.wikipedia.org/wiki/Animal)]</em>
</p>

The output of a classical SNN returns a single value:
* &nbsp;&nbsp;&nbsp;Same:    &emsp;Input samples are from the same class 
* Different:  &nbsp;Input samples are from different classes.

After the network has been tuned, the SNN provides good discriminative features. Due to this, the SNN can then be used to predict classes from new unseen data, but it can also be used on completely different, new classes, which have not been seen by the network during the training process.

One application for SNNs lies in the field of face recognition. For example, a system tracking the access right of people entering the building. Pure classification is not possible because the database only contains the faces of the recognized people who are allowed to enter the building. Especially due to the SNN learning how faces differ,  the database can be kept quite small.

### Starting Point
The main layer stack of the first implementation is based on the  [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) implementation.

<p align="center">
    <img src=Picture/Basicstructure.PNG width="700" >
</p>
<p align="center">
    <em> Figure 2: Starting point for SNN layer stack [source](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)  </em>
</p>

This first implementation makes use of the implemented [BCEloss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) function of Pytorch.

### Custom Loss Function
For the next implementation, a custom loss function is introduced. The mathematical foundation  of this implementation is based on the paper [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)

```Python
class ContrastiveLoss(torch.nn.Module):
	def __init__(self, margin=20.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin
	def forward(self, output1, output2, label):
		euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True ,p=1.0)
		loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2)
		                 + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
		return loss_contrastive
```
To use this, the SNN outputs the feature vectors instead of the calculated L1 norm.  With this custom loss function, the training time and impact could be increased.

###Classification Net
To improve the accuracy, a classification net was introduced. With three input vectors, the output gives back one of the two classes. 
=> This could increase the accuracy, but in this case, it does not.
&nbsp;=> Probably due to the poor feature extraction of the SNN.
&nbsp;&nbsp;=> Might be worth looking into as soon as reaching 70% accuracy.

###  Training on same slice numbers
Due to the data correlation between the different slices, the datasets cannot be shuffled randomly as before. The following procedure is used while loading the data:
1. Sorting data:  Sorting the data after person and slice number
2. Define sets: 20 slices are formed into one set/brain
3. Shuffle sets: Necessary to be able to split up train and validation set
4. Shuffle slices:  Same slice numbers are shuffled between sets (not used in the final implementation)

The following hints are important if implementing a dataset for the ADNI dataset:
 + Phyton function [glob()](https://github.com/python/cpython/blob/3.10/Lib/glob.py) returns image paths in random order
 + Inconsistent slice labeling in the dataset
  + Patient set 808819 slice 1 labeled with 88 (808819_88.jpeg)
  + Patient set 218391 slice 1 labeled with 78 (218391_78.jpeg)

### Using ResNet
The next step was implementing a ResNet approach for the siamese branch. The residual and identity blocks are built up from two convolutional layers with a skip connection. The convolutional blocks use a convolutional layer with stride=2 to downsample the input image, while possibly increasing the number of output channels. (=> [ResNet](https://arxiv.org/abs/1512.03385)). 

<p align="center">
    <img src=Picture/resnet.PNG width="600" >
</p>
<p align="center">
    <em> Figure 3: ResNet  </em>
</p>

The used ResNet uses an image input size of 105x105, therefore the slice images have to be resized from the original size before feeding into the neural net. 
=> Even though the network can prevent the same output for different inputs, the accuracy level stays at the same level.
### Using slices as channels

Maybe not all slices can be used for detecting the class. Therefore all 20 slices of a single person are packed into different channels of the tensor. This results in a sample size of 
20x240x256. 

### 3D Approach
To be able to use 3D convolutional layers, the slices have to be loaded as a third dimension and not as channels. This can easily be done by adding a dummy dimension.
```Python
torch.unsqueeze(image3D, dim=0)
```
3D convolutional layers are used to keep the information about the correlation of the different slices.

<p align="center">
    <img src=Picture/3Dslices.PNG width="300" >
</p>
<p align="center">
    <em> Figure 4: 3D data through stacking slices  </em>
</p>

### 3D Augmentation
Due to the correlation between the slices, the random augmentation from the previous implementation has to be modified. 

The used augmentation for randomly flipping the images vertically and horizontally have to be applied on all 20 slices at the same time. 

The same randomized crop and resize augmentation is applied to the two input samples of the two patients.

Furthermore, a randomized blackout augmentation was introduced which is applied individually on all 20 slices across the two input sampes.

<p align="center">
    <img src=Picture/augmentation.PNG width="600" >
</p>
<p align="center">
    <em> Figure 5: 3D-Augmentation shown on a single slice </em>
</p>

### Reducing the number of slices per patient
On the suggestion of a tutor, the number of used slices per person was reduced.
=> Improves training time 
=> Training loss is not consistently  decreasing but makes weird steps instead
=> Runs into the problem, same input =same output

-> Another Input from Tutor: You need all slices

###3D ResNet approach

The previously used ResNet approach was adapted to work with the 3D input data. The paper [Improving Deep Neural Network Interpretation for Neuroimaging Using Multivariate Modeling](https://link.springer.com/content/pdf/10.1007/s42979-022-01032-0.pdf) has acted as a guide for the 3D adaption of the implemented ResNet.

The final implementation of the 3D ResNet uses an input size of 210x210. The images only have to be cropped to the relevant area. For all brain scans the size of 210x210 is sufficient. Due to the preparation layer of the ResNet branch, the input does not have to be resized.

<p align="center">
    <img src=Picture/resnet3D.PNG width="600" >
</p>
<p align="center">
    <em> Figure 6:  3D ResNet   </em>
</p>

For the convolutional residual blocks 3 and 4, a stride with (2,1,1) is used, to decrease the number in the third dimension but not in the other two.

Tuning the batch size, and introducing a training step scheduler and gradient clipping, could increase the accuracy of the SNN. In this iteration step, these steps helped to reduce the training time to achieve the previous accuracy. 

Adding a deeper structure by using more residual blocks for the ResNet approach, didn't increase the accuracy. It only led to a drastic increase in training time and model size.

### One Shot image recognition
Due to the approach of a one-shot image recognition, only one sample person of two classes was used. Instead of using one reference image for both classes, the number could be increased to 10.
Averaging the slices across the patients does not provide an improvement. This is mainly due to the orientation and location of the brain centers being different between patients.

Averaging the slices itself across the patients, does not provide a improvement. This is mainly due to the orientation and location of the brain centers beeing different between patients.

<p align="center">
    <img src=Picture/singlemean.PNG width="600" >
</p>
<p align="center">
    <em> Figure 7: Averaged slice image (right) and single slice (left)  </em>
</p>

A more promising approach is to look at the average of the output feature vector of the SNN.

### Triplet Loss
A further improement is the introduction of triplet loss. With the help of triplet loss, the network can rank the similarity between the input instead of only labelling similar or different. The main difference to the contrastive loss is that it operates on three inputs.
* Anchor: Random sampe 
* Positive: Same class as anchor
* Negative: Different class

The euqation for the loss is defined as L=max(d(a,p)−d(a,n)+margin,0).

Minimizing the loss, the distance between the anchor and the positive sample d(a,p) is pushed to 0, while the distance bewteen the anchor and the negative sample d(a,n) converges to d(a,p)+margin.

<p align="center">
    <img src=Picture/tripletloss.PNG width="600" >
</p>
<p align="center">
    <em> Figure 8: Visualisation of driplet loss  </em>
</p>

After training, the recieved feature vector of the anchor should be more similar compared with the positve sample than the negative sample. 

###Adapting 3D ResNet approach

The previous introduced 3D ResNet, was modified which resulted in the final implementation.

<p align="center">
    <img src=Picture/resnet3D_final.PNG width="600" >
</p>
<p align="center">
    <em> Figure 8: Adapted 3D ResNet  </em>
</p>

The major change was the introduction of two additonal linear linear layers at the end of the branch structure and the removal of the previous used adaptive average pooling function. Furthermore the number of identiy blocks was reduced, so that all four stages contain 4 residual blocks. 
Especially due to the introduction of the average poling layer, the runtime of the SNN increased dramatically.

### Outlook:
Further improvements for the SNN implementation would be
* Reusing the additional classification network
* Tuning the SNN while training the classification Net
* Implement of true 3D augmentation (not slice based)

## Executing Code

The main code to train, validate and test the model is stored in train.py. To be able to run, the python files modules.py, dataset.py, and the ADNI dataset have to be in the same root folder as train.py. 
* Important constants are defined in the top section of the file 
* Predict.py loads the pre-trained model and reruns the testing on the test set 
* Plotted images are stored in the root folder 

## Results
The results are based on the last implementation of the SNN. 

<p align="center">
    <img src=Picture/training_loss.PNG width="600" >
</p>
<p align="center">
    <em> Figure 9: Loss during training epoch of training and validation set  </em>
</p>

The final accuracy could be improved but didnt peak the required 80%.

Summarising the main problems facing during the implementation:
* Different Dataset structures between training/validation and testing
* Labeling of Dataset
* Same output for all inputs
* Peak accuracy of 53% => equals distribution of the train set



## References
* https://adni.loni.usc.edu/
* https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
* https://en.wikipedia.org/wiki/Siamese_neural_network
* https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
* http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
* https://github.com/python/cpython/blob/3.10/Lib/glob.py
* https://arxiv.org/abs/1512.03385