# Image Classification
This algorithm Model.ipynb is tried to classify laterality (left or right sided knee) of the OAI AKOA knee data set having a minimum accuracy
of 0.9 on the test set. 

### Introduction
The Data set used for this algorithm is AKOA_Analysis. The algorithm separated the data set into three data sets, the test dataset, the training data set and the validation data set.

After separating the dataset, the algorithm starts to get the label of Image based on the name of each file. And divided data into two different groups, left and right. Also, the algorithm decreases the channel of images from three channels into one channel. 

The pictures below are the example of images after changed
![Image of Exmaples](https://github.com/theHughJin/PatternFlow/blob/master/recognition/S44301792/Image/Screen%20Shot%202020-11-07%20at%203.36.20%20PM.png)
       
The structure of the model is 
*  A 16 filters convolutional neural network which used ReLU function as the activation function 
*  A 32 filters convolutional neural network which used ReLU function as the activation function 
*  A flatten Layer
*  An output layer which used Softmax as the output function 
*The diagram below is the Summary of model
        ![Image of Summary of Model](https://github.com/theHughJin/PatternFlow/blob/master/recognition/S44301792/Image/Screen%20Shot%202020-11-08%20at%2011.54.59%20AM.png)        
Example code of the model is 
```python
       self.conv1 = tf.keras.layers.Conv2D(16, 1,input_shape=(256,256,1), activation = 'relu') 
       self.conv2 = tf.keras.layers.Conv2D(32, 1,input_shape=(256,256,1),activation = 'relu')
       self.flatten = tf.keras.layers.Flatten()
       self.outputLayer = tf.keras.layers.Dense(2,activation = 'softmax')
```
To better observe the accuracy of the training model. We increased the number of epochs.The figure below shows the accuracy and loss of the model's prediction on the training set and validation set during each training.
![Image of acc and loss](https://github.com/theHughJin/PatternFlow/blob/master/recognition/S44301792/Image/Screen%20Shot%202020-11-08%20at%201.10.21%20PM.png)   
The accuracy of the model on the final test set reached 100%
``` python 
       test_loss, test_acc = model.evaluate(test_ds.batch(8), verbose=2)
```
> 456/456 - 10s - loss: 1.2994e-05 - accuracy: 1.0000

### Prerequest
Tensorflow V2.0+ and Python V3.5+. Download the Glob and matplotlib library. Download Tensorflow [here](https://www.tensorflow.org/install)

## Appendix
### Data set 
OAI Acelerated Osteoarthritis knee data set (18K images) - This is part of the Osteoarthritis Initiative and comes with only labelled laterality (left/right knee labelling) in the [file](https://nda.nih.gov/oai/). The preprocessed version of this data set can be found on the course Blackboard site (under Course Help/Resources).
