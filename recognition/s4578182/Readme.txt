REPORT

Question 3: Segment the ISICs data set with the UNet [2] with all labels having a minimum Dice similarity coefficient
of 0.7 on the test set.

Dependencies:

1) Keras : Unet model function are from this package.
2) tensorflow-gpu: version 2.1.0 was installed. NVIDIA GeForce RTX 2080 was necessary for the Unex model to be fitted to the data
3) Scikit-learn : package was installed to split the data into train and test the data
4) Matplotlib: Used for plotting the image.
5) Pillow: to access image functions.

inputs:
Model used glob.glob function to import all the images from the folder.

Preprocessing steps:

1) Resizing the image into height=256, width=256
2) Normalising the image data:
	A) The train image ranges from 18 to 255 which was normalised within range 0 to 1 
	B) Ground truth data ranged from 0 to 255. Normalised to 0 or 1 with the condition, i) less than equal to 127 ii) greater than 127 equal to 1
3)Shape of the input :
	i) train dataset was converted into numpy and shape was attained to (2594, 256, 256, 3). Channel 3 means 3 physical elements to make the display.
	ii) Ground dataset was converted into numpy and shape was attained to (2594, 256, 256, 1)
4) Splitting the data using scikit-learn train_test_split() method. The test size is 33% of the dataset chosen at the random state = 42.
5) Building the Unet Model:

inputs_layer = Input(shape=(256,256,3))
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs_layer)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
merge6 = concatenate([drop4,up6], axis = 3)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
merge7 = concatenate([conv3,up7], axis = 3)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
merge8 = concatenate([conv2,up8], axis = 3)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
merge9 = concatenate([conv1,up9], axis = 3)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv10 =Conv2D(1,1, activation = 'sigmoid')(conv9)

model = Model(inputs =inputs_layer, outputs = conv10)

model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
The model takes in an input size of 256,256,3 and outputs as a convolution layer.

The contracting path has 3x3 convolution layer+ activation function 'relu'(kernel initializer='he_normal'), drop layer of probablity of 0.5, max pool layer of 2x2 of 4 blocks.
The Bottleneck contains convolution layer and dropout layer.
The expanding path contains 3x3 convolution layer and concatenation with the corresponding feature map.

The model uses sigmoid activation in the last convolution error. The loss function is binary crossentropy since it has single class with 2 classifying variables. The model uses 
Adam(lr = 1e-4) optimiser. 

6) Fitting the model:
epoch results:
Epoch 1/100
1737/1737 [==============================] - 78s 45ms/step - loss: 0.3649 - accuracy: 0.8448
Epoch 2/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.3028 - accuracy: 0.9040
Epoch 3/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.2825 - accuracy: 0.9138
Epoch 4/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.2636 - accuracy: 0.9244
Epoch 5/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.2570 - accuracy: 0.9263
Epoch 6/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.2515 - accuracy: 0.9305
Epoch 7/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.2515 - accuracy: 0.9300
Epoch 8/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.2438 - accuracy: 0.9326
Epoch 9/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.2379 - accuracy: 0.9360
Epoch 10/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.2371 - accuracy: 0.9358
Epoch 11/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.2296 - accuracy: 0.9379
Epoch 12/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.2295 - accuracy: 0.9391
Epoch 13/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.2287 - accuracy: 0.9387
Epoch 14/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.2186 - accuracy: 0.9433
Epoch 15/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.1885 - accuracy: 0.9390
Epoch 16/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.1497 - accuracy: 0.9416
Epoch 17/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.1447 - accuracy: 0.9436
Epoch 18/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.1395 - accuracy: 0.9454
Epoch 19/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.1387 - accuracy: 0.9448
Epoch 20/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.1292 - accuracy: 0.9481
Epoch 21/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.1282 - accuracy: 0.9499
Epoch 22/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.1214 - accuracy: 0.9512
Epoch 23/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.1277 - accuracy: 0.9495
Epoch 24/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.1134 - accuracy: 0.9540
Epoch 25/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.1157 - accuracy: 0.9535
Epoch 26/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.1035 - accuracy: 0.9578
Epoch 27/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.1013 - accuracy: 0.9593
Epoch 28/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0987 - accuracy: 0.9599
Epoch 29/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0927 - accuracy: 0.9627
Epoch 30/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0889 - accuracy: 0.9636
Epoch 31/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0874 - accuracy: 0.9645
Epoch 32/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0891 - accuracy: 0.9643
Epoch 33/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0899 - accuracy: 0.9640
Epoch 34/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0752 - accuracy: 0.9692
Epoch 35/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0683 - accuracy: 0.9717
Epoch 36/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0661 - accuracy: 0.9724
Epoch 37/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0647 - accuracy: 0.9731
Epoch 38/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0587 - accuracy: 0.9753
Epoch 39/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0604 - accuracy: 0.9749
Epoch 40/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0550 - accuracy: 0.9768
Epoch 41/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0557 - accuracy: 0.9766
Epoch 42/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0506 - accuracy: 0.9785
Epoch 43/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0471 - accuracy: 0.9799
Epoch 44/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0471 - accuracy: 0.9799
Epoch 45/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0448 - accuracy: 0.9809
Epoch 46/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0452 - accuracy: 0.9809
Epoch 47/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0478 - accuracy: 0.9798
Epoch 48/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0493 - accuracy: 0.9792
Epoch 49/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0565 - accuracy: 0.9770
Epoch 50/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0447 - accuracy: 0.9813
Epoch 51/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0391 - accuracy: 0.9834
Epoch 52/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0376 - accuracy: 0.9840
Epoch 53/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0363 - accuracy: 0.9845
Epoch 54/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0348 - accuracy: 0.9850
Epoch 55/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0340 - accuracy: 0.9854
Epoch 56/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0342 - accuracy: 0.9854
Epoch 57/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0325 - accuracy: 0.9861
Epoch 58/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0322 - accuracy: 0.9862
Epoch 59/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0328 - accuracy: 0.9860
Epoch 60/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0324 - accuracy: 0.9862
Epoch 61/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0314 - accuracy: 0.9866
Epoch 62/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0345 - accuracy: 0.9855
Epoch 63/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0411 - accuracy: 0.9832
Epoch 64/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0371 - accuracy: 0.9845
Epoch 65/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0332 - accuracy: 0.9860
Epoch 66/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0285 - accuracy: 0.9878
Epoch 67/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0269 - accuracy: 0.9885
Epoch 68/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0260 - accuracy: 0.9888
Epoch 69/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0253 - accuracy: 0.9892
Epoch 70/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0252 - accuracy: 0.9892
Epoch 71/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0244 - accuracy: 0.9896
Epoch 72/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0449 - accuracy: 0.9820
Epoch 73/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0835 - accuracy: 0.9687
Epoch 74/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0316 - accuracy: 0.9868
Epoch 75/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0263 - accuracy: 0.9889
Epoch 76/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0238 - accuracy: 0.9898
Epoch 77/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0227 - accuracy: 0.9903
Epoch 78/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0223 - accuracy: 0.9905
Epoch 79/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0221 - accuracy: 0.9905
Epoch 80/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0215 - accuracy: 0.9908
Epoch 81/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0214 - accuracy: 0.9908
Epoch 82/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0209 - accuracy: 0.9910
Epoch 83/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0215 - accuracy: 0.9908
Epoch 84/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0208 - accuracy: 0.9911
Epoch 85/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0203 - accuracy: 0.9913
Epoch 86/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0198 - accuracy: 0.9915
Epoch 87/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0206 - accuracy: 0.9912
Epoch 88/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0212 - accuracy: 0.9910
Epoch 89/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0199 - accuracy: 0.9915
Epoch 90/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0194 - accuracy: 0.9917
Epoch 91/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0189 - accuracy: 0.9919
Epoch 92/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0197 - accuracy: 0.9916
Epoch 93/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0215 - accuracy: 0.9911
Epoch 94/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0195 - accuracy: 0.9916
Epoch 95/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0185 - accuracy: 0.9921
Epoch 96/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0178 - accuracy: 0.9924
Epoch 97/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0179 - accuracy: 0.9923
Epoch 98/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0495 - accuracy: 0.9813
Epoch 99/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0346 - accuracy: 0.9861
Epoch 100/100
1737/1737 [==============================] - 68s 39ms/step - loss: 0.0207 - accuracy: 0.9912

Dice Similarity Coefficient:0.8325091462746881

Comparison with Research Paper:
H-DenseUNet: Hybrid Densely Connected UNet for Liver and Tumor Segmentation From CT Volumes
Authors: Xiaomeng Li; Hao Chen; Xiaojuan Qi; Qi Dou; Chi-Wing Fu; Pheng-Ann Heng

The variation from my model to research paper is that the max pool uses 3x3 maxpool layer where
as my model uses 2x2 for better performance. Their model used various efficiency model to test the 
TEST dataset, while mine used dice similarity coefficent. Their model had many classes, and used 
softmax activation with sparse categorical loss function while mine used sigmoid activation 
function with binary crossentropy as loss function.
 