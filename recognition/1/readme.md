#Topic: segment the OASIS brain dataset with improved UNet
Name: zhengzhao li
student number: 45542976

#Description
The segmentation of brain tumors is critical to the clinic, and manual segmentation is very cumbersome, so I have worked hard in developing segmentation algorithms form of Convolutional Neural Network My network architecture is inspired by the U-Net and has been carefully modified to maximize the performance of brain tumor segmentation. I use dice loss function Cope with class imbalance and use extensive data augmentation to successfully prevent overfitting.

#How it works
I divided this project into five parts. The first part is to create a data loader to load the data and images I need. The second part is to create a model, I create convolutional layer according to the model given by the pdf, and the dice similarity coefficient with an accuracy of 0.9. The third part is model training, I plot the training and validation loss and accuracy. The fourth part is the visual prediction. I will first load the data and then print out the image/prediction/ground truth of the data.  and the fifth part is Get the segmentation metrics, I randomly select an image and Print out his prediction and ground truth.
