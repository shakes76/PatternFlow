from dataset import DataProcess
import matplotlib.pyplot as plt
TRAINDATA,VALIDDATA,TESTDATA=DataProcess()
import dill as pickle
with open('goodmodel2.pkl', 'wb') as file:
 finalmodel=pickle.load(file)
predict=finalmodel((TESTDATA[t].cuda().float().reshape((1,TESTDATA[t].shape[0],TESTDATA[t].shape[1],TESTDATA[t].shape[2])))[0].cpu().detach().numpy()) 
for i in range(0,int(len(TESTDATA)/4)):
  ssimtest=metric(torch.tensor(np.float32(predict[range(i*4,(i+1)*4),:])),TESTDATA.to(torch.float32)[range(i*4,(i+1)*4),:])+ssim
 ssimtest=ssimtest/len(TESTDATA)
print(ssimtest)# The mean ssim on the predictions of the best model on the test set
#testing on test data set to check for clear image
plt.imshow(predict[0,:,:,:].astype('float32'),cmap="Greys_r")
plt.imshow(predict[1,:,:,:].astype('float32'),cmap="Greys_r")  
plt.imshow(predict[2,:,:,:].astype('float32'),cmap="Greys_r")
plt.imshow(predict[3,:,:,:].astype('float32'),cmap="Greys_r")
plt.imshow(predict[4,:,:,:].astype('float32'),cmap="Greys_r")