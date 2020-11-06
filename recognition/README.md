# Recognition Problems
# Classify laterality (left or right sided knee) of the OAI AKOA knee data set
* Student Name: Xu Jianliang
* Student No. 45855537
* Student E-mail: s4585553@student.uq.edu.au

## File specification
* dataProcess.py is the file to Process the raw data, create two folders and write photos of the left knee and the right knee into the two folders respectively.
* train.py is the file to train the dataset，here we use DenseNet as base net, and the weights of ImageNet, then we train the last seven layers.
* predict.py is the file to predict the results of the test data.

## Requirment:
Version: Tensorflow >=2.20

## Algorithm:
### DenseNet
![image](https://github.com/Alexu0506/PatternFlow/blob/topic-recognition/recognition/p1.png)
* 5-layer dense block with a growth rate of k=4. Each layer takes all preceding feature-maps as input.  

List several advantages of densenet:
1. Reduced vanishing gradient
2. Enhanced feature delivery
3. More effective use of features
4. To some extent, the number of parameters is reduced

In the deep learning network, with the deepening of networkdepth, the problem of gradient disappearance will become more and more obvious.
At present, many papers have proposed solutions to this problem, such as RESNET, highway networks, stochastic depth, fractalnetworks, etc.
Although the net work structure of these algorithms is defferent, the core is: create short paths from early layers to later layers.
So what does the author do? Continue this idea, that is to ensure the maximum information transmission betweenthe middle layer and layer of the network,
directly connect all layers!

First put a structure diagram of the dent block. In traditional convolutional neural networks, if you have an L-layer, there will be l connections, but in densenet, there will be l (L + 1) / 2 connections. In short, the input of each layer comes from the output of all previous layers. As shown in the following figure: x0 is the input, H1 is x0 (input), H2 is x0 and X1 (x1 is the output of H1)……
One advantage of densenet is that the network is narrower and has fewer parameters, which is largely due to the design of this dense block. It is mentioned later that the number of output feature maps of each volume layer in dense block is very small (less than 100), instead of being as wide as hundreds or thousands of other networks. At the same time, this connection makes the transmission of features and gradients more effective, and the network is easier to train. Each layer has direct access to the gradients from the loss function and the original input signal, leading to an implicit deep supervision. As mentioned above, the gradient disappearance problem is more likely to occur when the depth of the network is deeper. The reason is that the input information and gradient information are transmitted between many layers. Now, this kind of dense connection is equivalent to that each layer directly connects input and loss. Therefore, the phenomenon of gradient disappearance can be reduced, so the deeper network is not a problem. In addition, the author also observed that this kind of dense connection has regularization effect, so it has a certain inhibitory effect on over fitting.

## Model architecture
![image](https://github.com/Alexu0506/PatternFlow/blob/topic-recognition/recognition/p2.png)

## Train result
![image](https://github.com/Alexu0506/PatternFlow/blob/topic-recognition/recognition/p3.png)
