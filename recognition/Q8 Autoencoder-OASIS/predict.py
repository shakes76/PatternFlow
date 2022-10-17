def Predict():
 """This function runs the best model chosen train.py and uses it to output the average SSIM of this model on the testdata and some test images to show that they are clear

    Input: None

    Output: None
 """   
    
 from dataset import DataProcess
 import matplotlib.pyplot as plt
 TRAINDATA,VALIDDATA,TESTDATA=DataProcess()
 TESTDATA=TESTDATA.permute(0,3,2,1).contiguous() #Reshaping Data to be of form: Image Number*channel*height*width*channel
 import dill as pickle
 with open('goodmodel2.pkl', 'wb') as file:
  finalmodel=pickle.load("put path of file containing model here") #This loads the model which performed best on the validation data
 predict=finalmodel((TESTDATA.cuda().float().reshape((1,TESTDATA[t].shape[0],TESTDATA[t].shape[1],TESTDATA[t].shape[2])))[0].cpu().detach().numpy()) #Computes the predictions/reconstructed images on the testdata

 for i in range(0,int(len(TESTDATA)/4)): #issues may arise trying to compute SSIM for predicted images versus the original images in the test data set, all at once so the SSIM sums are calculated piecemeal
  ssimtest=metric(torch.tensor(np.float32(predict[range(i*4,(i+1)*4),:])),TESTDATA.to(torch.float32)[range(i*4,(i+1)*4),:])+ssim
 ssimtest=ssimtest/len(TESTDATA)

 predict=predict.permute(0,3,2,1) #Reshaping predicted images to be of form Image Number*height*width*channel
 plt.imshow(predict[0,:,:,:].astype('float32'),cmap="Greys_r") #testing on test data set to check for clear image
 plt.show()
 plt.imshow(predict[1,:,:,:].astype('float32'),cmap="Greys_r")  
 plt.show()
 plt.imshow(predict[2,:,:,:].astype('float32'),cmap="Greys_r")
 plt.show()
 plt.imshow(predict[3,:,:,:].astype('float32'),cmap="Greys_r")
 plt.show()
 plt.imshow(predict[4,:,:,:].astype('float32'),cmap="Greys_r")
print(ssimtest)# The mean ssim on the predictions of the best model on the test set