from preprocesseddata import *
from unetImproved import *
from dicfunction import *
from test import *
import glob
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# load image ISIC image
isic_input = glob.glob("D:/2021S2/COMP3710/ass/report/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2/*.jpg")
isic_ground_truth = glob.glob("D:/2021S2/COMP3710/ass/report/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2/*.png")
print(len(isic_input))

#use mnist first
#(x_train, y_train), (x_test, y_test) = mnist.load_data()



#process image
X = preprocess_array(isic_input)/ 255.
y = np.round(preprocess_array_truth(isic_ground_truth)/ 255)

#split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#fit unetmodel
epoch_size=100
batch=33
untmodel = unetmodel()
fit(unetmodel,x_train,y_train,epoch_size,batch)

#predict
pred = np.round(unetmodel.predict(x_test,batch_size=33))

#the dice coefficient
dice = diceCoefficient(y_test, pred,1.)

#display the predict result
for i in range(9):
    plotResult([x_train[i],y_train[i],pred[i]])










