#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# We import the image data and the corresponding ground truth segmentation images. 
input_data = next(os.walk('Downloads/ISIC2018_Task1-2_Training_Input_x2'))[2]
truth_data = next(os.walk('Downloads/ISIC2018_Task1_Training_GroundTruth_x2'))[2]
input_data.sort()
truth_data.sort()

#Resize image to 256x256 and create two array for preparation of dataset spliting.
data = []
label = []

for i in input_data:    
    if(i.split('.')[-1]=='jpg'):                      
        img = cv2.imread('Downloads/ISIC2018_Task1-2_Training_Input_x2/{}'.format(i), cv2.IMREAD_COLOR)
        resized_img = cv2.resize(img,(256, 256), interpolation = cv2.INTER_CUBIC)
        
        data.append(resized_img)
        
        truth = cv2.imread('Downloads/ISIC2018_Task1_Training_GroundTruth_x2/{}'.format(i.split('.')[0]+'_segmentation.png'), cv2.IMREAD_GRAYSCALE)
        resized_truth = cv2.resize(truth,(256, 256), interpolation = cv2.INTER_CUBIC)
        
        label.append(resized_truth

# Spliting data into Training dataset and Test dataset
# The rate of training dataset/test dataset is 8/2. Also, we need to normalize data(divided by 255).                     
data = np.array(data)
label = np.array(label)

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2, random_state=3)

Y_train = Y_train.reshape((Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1))
Y_test = Y_test.reshape((Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],1))

X_train = X_train / 255
X_test = X_test / 255
Y_train = Y_train / 255
Y_test = Y_test / 255

Y_train = np.round(Y_train,0)
Y_test = np.round(Y_test,0)

# Write function to test the m Dice similarity coefficient of model.
def dice_coef(y_true, y_pred):
    smooth = 0.0
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    y_pred = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Training model on training dataset and truth dataset.
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[dice_coef])
history = model.fit(X_train, Y_train, batch_size=16, 
          epochs=15, verbose=1,validation_data=(X_test,Y_test))

# Then, we can compare the origin image, ground truth and the predict image.
preds_test = model.predict(X_test, verbose=1)
plt.subplot(1, 3, 1)
plt.title('origin')
plt.imshow(X_test[5])

plt.subplot(1, 3, 2)
plt.title('ground truth')
plt.imshow(Y_test[5],cmap='gray')

plt.subplot(1, 3, 3)
plt.title('predict')
plt.imshow(preds_img[5],cmap='gray')

plt.show()

