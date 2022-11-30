def Predict():
 """This function runs the best model chosen train.py and uses it to output the average SSIM of this model on the testdata and some test images to show that they are clear

    Input: None

    Output: None
 """   
    
 from dataset import DataProcess
 import matplotlib.pyplot as plt
 import numpy as np
 from torchmetrics import StructuralSimilarityIndexMeasure
 metric =StructuralSimilarityIndexMeasure(data_range=1.0,reduction='sum')
 import numpy as np
 import torch
 TRAINDATA,VALIDDATA,TESTDATA,DATALOADER=DataProcess()
 torch.manual_seed(0)
 predict=np.empty((len(TESTDATA),TESTDATA.shape[1],TESTDATA.shape[2],TESTDATA.shape[3]))
 #TESTDATA=TESTDATA.permute(0,3,2,1).contiguous() #Reshaping Data to be of form: Image Number*channel*height*width*channel
 import dill as pickle
 with open('path and finalname of final model', 'rb') as file:
  finalmodel=pickle.load(file) #This loads the model which performed best on the validation data
 for t in range(0,len(TESTDATA)):
   
  predict[t,:,:,:]=finalmodel(TESTDATA[t].cuda().float().reshape((1,TESTDATA[t].shape[0],TESTDATA[t].shape[1],TESTDATA[t].shape[2])))[0].cpu().detach().numpy() #Computes the predictions/reconstructed images on the testdata
 torch.manual_seed(0) #resetting the seed so that the test SSIM equals the value produced by predict.py
 ssimtest=0
 for i in range(0,int(len(TESTDATA)/4)): #issues may arise trying to compute SSIM for predicted images versus the original images in the test data set, all at once so the SSIM sums are calculated piecemeal
  ssimtest=metric(torch.tensor(np.float32(predict[range(i*4,(i+1)*4),:,:,:])),TESTDATA.to(torch.float32)[range(i*4,(i+1)*4),:,:,:])+ssimtest
 ssimtest=ssimtest/len(TESTDATA)

 predict=torch.tensor(predict).permute(0,3,2,1).detach().numpy() #Reshaping predicted images to be of form Image Number*height*width*channel
 plt.imshow(predict[0,:,:,:].astype('float32'),cmap="Greys_r") #testing on test data set to check for clear image
 plt.show()
 plt.imshow(predict[1,:,:,:].astype('float32'),cmap="Greys_r")  
 plt.show()
 plt.imshow(predict[2,:,:,:].astype('float32'),cmap="Greys_r")
 plt.show()
 plt.imshow(predict[3,:,:,:].astype('float32'),cmap="Greys_r")
 plt.show()
 plt.imshow(predict[4,:,:,:].astype('float32'),cmap="Greys_r")
 plt.show()
 print(f'The best model has a mean ssim on the test set of {ssimtest}')
 
# The mean ssim on the predictions of the best model on the test set
def GeneratedImages():
  """This function generates new images using the Priormodel saved in 'PredictPrior' in 'train.py' and the best VQVAE saved in 'TRAINVALTEST' in 'train.py'

    Input: None

    Output: None
 """ 
  from dataset import DataProcess
  from dataset import dataencodings
  import dill as pickle
  import torch
  import torch.nn.functional as F
  import numpy as np
  import matplotlib.pyplot as plt
  with open('Priormodel', 'rb') as file: #Loading the CNN model that generates the prior probabilities of each embedding vector for each sample and pixel
   priormodel=pickle.load(file)
  with open('path and finalname of trained prior model', 'rb') as file: #Loading the best VQVAE model
   finalmodel=pickle.load(file)  
  def Sampler(data,priormodel):
   
   out=priormodel(data.cuda().float()) #Run the CNN (for prior probabilities) on the input
   out=F.softmax(out,dim=1)            #Convert the output of the CNN model to probabilities
   #print(out.shape)
   #out=out.permute(0,3,2,1)
   #out=np.mean(out, -1)
   from torch.distributions import Categorical
   return Categorical(out.permute(0,3,2,1)).sample() #Generate  arrays containing embedding vector indexes for each sample and pixel of the input using the probabilities output from the softmax function

  No=5 #number of images to generate
 
  samplearray=np.zeros((No,256,256))
  height=256
  width=256
  for x in range(0,height):  #The purpose of these two loops is to generate indexes for each sample and pixel, one at time, as a result of conditionalizes the index probability of a pixel on the indexes generated for previous pixels
   
   for y in range(0,width):
      sample=Sampler(torch.tensor(samplearray),priormodel)
      samplearray[:,x,y]=sample[:,x,y].cpu().detach().numpy()
    
  embeddingarray=finalmodel.embedding.weight
  embeddingsamples=F.one_hot(torch.tensor(samplearray).long(),finalmodel.numembedding).float() #convert the generated embedding vector indexes into one hot encoded format
  
  embeddingvectorsamples=torch.matmul(embeddingsamples.cuda(),embeddingarray.T) #Use the one-hot encoded embedding vector indexes to select the embedding vectors corresponding to those indexes    
  img=F.sigmoid(finalmodel.layer6(finalmodel.layer5(F.relu(finalmodel.layer4(embeddingvectorsamples.permute(0,3,2,1)))))).cpu().detach().numpy() #Run on the decoder from the best VQVAE model on the embedding vectors generated for each sample and pixel, selected in the previous line
  img=torch.tensor(img).permute(0,3,2,1).detach().numpy() #permute the image arrays into the right dimensional format so they can be viewed
  plt.imshow(img[0,:,:,:].astype('float32'),cmap="Greys_r") #viewing the generated images to check for clear image
  plt.show()
  plt.imshow(img[1,:,:,:].astype('float32'),cmap="Greys_r")  
  plt.show()
  plt.imshow(img[2,:,:,:].astype('float32'),cmap="Greys_r")
  plt.show()
  plt.imshow(img[3,:,:,:].astype('float32'),cmap="Greys_r")
  plt.show()
  plt.imshow(img[4,:,:,:].astype('float32'),cmap="Greys_r")
  plt.show()