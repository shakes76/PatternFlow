
def TRAINVALTEST():
 """Function used to iterate through various hyperparameters, train the associated model, and return the optimal model

    Inputs: None

    Outputs: 
           mod:
             This is the optimal model
           ssim: numpy array
              This is an array containing the average SSIM values over the validation dataset for each hyperparameter combination
           ssimtest: Float
              This is the average SSIM over the test dataset for the combination of hyperparameters that achieves the highest/best average SSIM over the validation dataset 
"""
 import modules
 from modules import VQVAE1
 from dataset import DataProcess
 from pandas import DataFrame
 import numpy as np
 from torchmetrics import StructuralSimilarityIndexMeasure
 import torch   
 TRAINDATA,VALIDDATA,TESTDATA, DATALOADER=DataProcess() #Loading and augmenting the training data, validation data and test data
 #TRAINDATA=TRAINDATA[0:125,:,:,:]
 #VALIDDATA=VALIDDATA[0:120,:,:,:]
 

 DimLatSpace=np.array([10,20,40]) #The values of the latent space dimensionality that are iterated over
 Numberofembeddings=np.array([10,100]) #The values of 'the number of embeddings' that are iterated over
 learningrate=np.array([0.00002,0.0001,0.001]) #The values of the learning rate that are iterated over
 commitcost=np.array([0.1,0.3,0.6])             #The values of the commitment cost that are iterated over
 mod=[]              #The array used to store the each model, where each model corresponds to a distinct hyperpameter combinations
 predict=np.empty((len(VALIDDATA),VALIDDATA.shape[1],VALIDDATA.shape[2],VALIDDATA.shape[3])) #array used to store the predictions of a model
 metric =StructuralSimilarityIndexMeasure(data_range=1.0,reduction='sum')  #setting up the SSIM metric
 ssim=np.empty((3,2,3,3))  #3,2,3,3                                                #array used to store the validation SSIM for each model
 torch.manual_seed(0)
 for i in range(0,3):                                           #iterating over the hyperparameters
    for j in range(0,2):
        for z in range(0,3):
            for k in range(0,3):
             mod.append(VQVAE1(TRAINDATA,DATALOADER,DimLatSpace[i],Numberofembeddings[j],learningrate[z],commitcost[k]))
             
             for t in range(0,len(VALIDDATA)): #Predicting/reconstructing the images using the validation data set
              predict[t,:,:,:]=mod[18*i+9*j+3*z+k](VALIDDATA[t].cuda().float().reshape((1,VALIDDATA[t].shape[0],VALIDDATA[t].shape[1],VALIDDATA[t].shape[2])))[0].cpu().detach().numpy()
              #Predictions were made, one at a time, according to the line above due to memory issues that result if you try make predictions for the entire Validation dataset all at once
             ssim[i,j,z,k]=0
             for g in range(0,len(VALIDDATA/10)): #Computing the SSIM metric for each hyperparameter combination
               ssim[i,j,z,k]=metric(torch.tensor(np.float32(predict[range(i*10,(i+1)*10),:,:,:])),VALIDDATA.to(torch.float32)[range(i*10,(i+1)*10),:,:,:])+ssim[i,j,z,k]
             ssim[i,j,z,k]=ssim[i,j,z,k]/len(VALIDDATA) #issues may arise trying to compute SSIM for entire dataset all at once so the SSIM sums are calculated piecemeal
 maxssim=np.max(ssim)   #This finds the value of the best ssim
 indices=np.where(ssim==maxssim)#Finds the indices (and thus, the hyperparameter combination) associated with the best/biggest SSIM
 predict=np.empty((len(TESTDATA),TESTDATA.shape[1],TESTDATA.shape[2],TESTDATA.shape[3]))
 for t in range(0,len(TESTDATA)):  
  predict[t,:,:,:]=mod[18*indices[0][0]+9*indices[1][0]+3*indices[2][0]+indices[3][0]](TESTDATA[t].cuda().float().reshape((1,TESTDATA[t].shape[0],TESTDATA[t].shape[1],TESTDATA[t].shape[2])))[0].cpu().detach().numpy()
 ssimtest=0
 for i in range(0,int(len(TESTDATA)/4)):  #Computing the SSIM metric of the best model over the test data set
  ssimtest=metric(torch.tensor(np.float32(predict[range(i*4,(i+1)*4),:,:,:])),TESTDATA.to(torch.float32)[range(i*4,(i+1)*4),:,:,:])+ssimtest #issues may arise trying to compute SSIM for entire dataset all at once so the SSIM sums are calculated piecemeal
 ssimtest=ssimtest/len(TESTDATA) #Finds average SSIM of the best model over the testdata
 import dill as pickle
 with open('goodmodel2.pkl', 'wb') as file:  #Saves the best model
  pickle.dump(mod[18*indices[0][0]+9*indices[1][0]+3*indices[2][0]+indices[3][0]], file)
 return mod,ssim,ssimtest,indices             