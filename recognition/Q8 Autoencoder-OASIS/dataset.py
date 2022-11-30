def DataProcess():
 """Function used to load the data, perform data augmentation, and create a dataloader

     Inputs:None

     Outputs:
       TRAINDATA: This is the augmented (includes images that are flipped upside down) Training dataset
       VALIDDATA: This is the validation data set
       TESTDATA:  This is the TESTDATA
       DATALOADER: This is the data loader constructed over the training dataset
 """
 from torch.utils.data.sampler import SubsetRandomSampler
 from torch.utils.data import DataLoader   
 import numpy as np
 import torch
 import torchvision.transforms as transforms
 from PIL import Image
 torch.manual_seed(0)
 TRAINDATA=np.empty((9664*2,256,256,3),dtype='float16') #initializing array to hold Traindata- this includes the original Traindata as well as the Traindata flipped upside down. It is in Float 16 due to memory issues.
 TRAINDATA=torch.tensor(TRAINDATA)
 
 for i in range(0,9664):       #This loop ranges across the training data and loads the images. An index j is used to record the number corresponding to the current block of 32 images As, the images are present in numbered blocks of 32 images, the if statements are used to skip to the next number, if there is no block of images corresponding to the current number j
    j=int(np.floor(i/32)+1) #This sets the block number 
    if j>=8:
        j=j+1
    if j>=24:
        j=j+1
    if j>=36:
        j=j+1
    if j>=48:
        j=j+1
    if j>=73:
        j=j+8
    if j>=89:
        j=j+1
    if j>=92:
        j=j+59
    if j>=154:
        j=j+1
    if j>=171:
        j=j+2 
    if j>=175:
        j=j+1 
    if j>=187:
        j=j+1
    if j>=194:
        j=j+1
    if j>=196:
        j=j+1
    if j>=215:
        j=j+1
    if j>=219:
        j=j+1
    if j>=225:
        j=j+1
    if j>=242:
        j=j+1
    if j>=245:
        j=j+1
    if j>=248:
        j=j+1
    if j>=251:
        j=j+2
    if j>=257:
        j=j+1
    if j>=276:
        j=j+1
    if j>=297:
        j=j+1
    if j>=306:
        j=j+1
    if j>=320:
        j=j+1
    if j>=324:
        j=j+1
    if j>=334:
        j=j+1
    if j>=347:
        j=j+1
    if j>=360:
        j=j+1
    if j>=364:
        j=j+1
    if j>=391:
        j=j+1
    if j>=393:
        j=j+1

                            


    h=f'{j}'   #h here is presenting the number j, corresponding to  which numbered block of 32 images is being dealt with, in the way that is written in the filename so that the image can be loaded
    if j<10:
        h=f'00{j}'
    if j<100 and j>=10:
        h=f'0{j}'    
    z=int(i-32*np.floor(i/32)) # z determines the current image number within the current block of 32 images
    img = Image.open(f'path of folder containing train data/case_{h}_slice_{z}.nii.png').convert('RGB') #loads the image
    img=np.array(img) #convert the image into an array
    #to_tensor = transforms.ToTensor()
    #tensor = to_tensor(img)
    tensor=torch.tensor(img)/255 #converting the array so that pixels assume values between 0 and 1
    TRAINDATA[i,:,:,:]=tensor.float()
    TRAINDATA[9664+i,:,:,:]=torch.flipud(tensor).float() #adding the flipped version of the image to the dataset
 TRAINDATA=TRAINDATA.float()
 TRAINDATA=TRAINDATA.permute(0, 3, 2, 1).contiguous() #This re-arranges the train dataset into the form: number of images*channels*height*width 
 
 num_workers =0
 batch_size = 25 #setting Batch size
 num_train = len(TRAINDATA)
 
 indices = list(range(num_train))

 np.random.shuffle(indices) #randomly shuffling the data

 train_index= indices

 train_sampler = SubsetRandomSampler(train_index) #creating sampler for the training data

 DATALOADER = torch.utils.data.DataLoader(TRAINDATA, batch_size = batch_size,
                                           sampler = train_sampler, num_workers = num_workers) #setting up the data loader
    
 import torchvision.transforms as transforms
 from PIL import Image
 VALIDDATA=np.empty((1120,256,256,3),dtype='float16') #initializing array to hold Training data. It is in Float 16 due to memory issues
 VALIDDATA=torch.tensor(VALIDDATA)
 for i in range(0,1120): #This loop ranges across the validation data and loads the images. An index j is used to record the number corresponding to the current block of 32 images As, the images are present in numbered blocks of 32 images, the if statements are used to skip to the next number, if there is no block of images corresponding to the current number j
     
    j=int(np.floor(i/32)+402) #This sets the block number 
    #print(j)
    if j>=412:
        j=j+1
    if j>=414:
        j=j+1
    if j>=427:
        j=j+1
    if j>=436:
        j=j+1
    
    h=f'{j}'       #h here is presenting the number j, corresponding to  which numbered block of 32 images is being dealt with.
    z=int(i-32*np.floor(i/32)) # z determines the current image number within the current block of 32 images
    img = Image.open(f'path of folder containing validation data/case_{h}_slice_{z}.nii.png').convert('RGB') #loads the image
    img=np.array(img) #convert the image into an array
   
  
    tensor=torch.tensor(img)/255 #converting the array so that pixels assume values between 0 and 1
    VALIDDATA[i,:,:,:]=tensor 
 VALIDDATA=VALIDDATA.permute(0, 3, 2, 1).contiguous() #This re-arranges the validation dataset into the form: number of images*channels*height*width 
 
 import torchvision.transforms as transforms
 from PIL import Image
 TESTDATA=np.empty((544,256,256,3),dtype='float16') #initializing array to hold Testdata.It is in Float 16 due to memory issues.
 TESTDATA=torch.tensor(TESTDATA)
 for i in range(0,544): #This loop ranges across the test data and loads the images. An index j is used to record the number corresponding to the current block of 32 images. 
    j=int(np.floor(i/32)+441) #This sets the block number 
          

   
    
    h=f'{j}'       #h here is presenting the number j, corresponding to  which numbered block of 32 images is being dealt with.
    z=int(i-32*np.floor(i/32)) # z determines the current image number within the current block of 32 images
    img = Image.open(f'path of folder containing test data/case_{h}_slice_{z}.nii.png').convert('RGB') #loads the image
    img=np.array(img) #convert the image into an array
    
    tensor=torch.tensor(img)/255 #converting the array so that pixels assume values between 0 and 1
    TESTDATA[i,:,:,:]=tensor
 TESTDATA=TESTDATA.permute(0,3,2,1).contiguous()  #This re-arranges the test dataset into the form: number of images*channels*height*width 
 return TRAINDATA,VALIDDATA,TESTDATA,DATALOADER 


def dataencodings(DATA):
 """This function extracts the encodings from the best saved VQVAE model and feeds them into a dataloader

     Input:
         DATA (torch tensor_): The Training Data used to train the best model using VQVAE1. This is the output, 'TRAINDATA' from the 'DataProcess' function
          
         

     Returns:
         Encoding: The embedding vector indexes, specifying for each pixel of each sample of TRAINDATA, an index which determines which embedding vector corresponds to that sample and pixel according to best saved VQVAE model
         dataloader: This is a dataloader that the 'Encoding' array has been fed into which is required to use 'PriorCNN'
 """   
 import torch
 import dill as pickle
 import numpy as np
 from torch.utils.data.sampler import SubsetRandomSampler
 from torch.utils.data import DataLoader
 import torch.nn.functional as F   
 with open('path and finalname of final model', 'rb') as file: #loading the best VQVAE model found by running 'TRAINVALTEST' in 'train.py'
  finalmodel=pickle.load(file)
  

 
 encoding=np.empty((len(DATA),DATA.shape[2],DATA.shape[3])) #initializing the array to hold the encodings (indexes of embedding vectors for each sample and pixel of DATA) generated by the best VQVAE model
 

 for t in range(0,len(DATA)): #This loops generates the encoding (index of each embedding vector) of each training sample's pixels. It is done one sample at a time due to memory issues. Encoding is a result of running first the encoder and then the VQVAE layer
  
  x=finalmodel.VQVAE(finalmodel.layer3(F.relu(finalmodel.layer2(finalmodel.layer1(F.relu(finalmodel.layer0(DATA[t].cuda().float().reshape((1,DATA[t].shape[0],DATA[t].shape[1],DATA[t].shape[2])))))))),finalmodel.numembedding,finalmodel.embeddingdim,finalmodel.commitcost)[2].cpu().detach().numpy()
  x=x.reshape((256,256)) 
  encoding[t,:,:]=x
   
 
 encoding=torch.tensor(encoding)  # This part is used to generate a dataloader for the encodings of the dataset
 num_workers =0
 batch_size = 15
 num_encoding = len(encoding)
 
 indices = list(range(num_encoding))

 np.random.shuffle(indices) #randomly shuffling the data

 index= indices

 sampler = SubsetRandomSampler(index) #creating sampler for the encoding data

 DATALOADER = torch.utils.data.DataLoader(encoding, batch_size = batch_size,
                                           sampler = sampler, num_workers = num_workers) #The dataloader for the encoding data
 return encoding, DATALOADER  #print(i)


  #print(x.shape)ra
  
  #for i in range(0,finalmodel.numembedding):
    #print(x[0,:,:,:]==embed[i,:])
   #m=np.where(np.all(x==embed[i,:],axis=1))
   #encoding[([t]*len(m[0]),m[0],m[1])]=i
 #for i in range(0,TRAINDATA.shape[2]):
  #  for j in range(0,TRAINDATA.shape[3]):
        #print(np.all(()==x[0,:,i,j],axis=1))
   #     encoding[t,i,j]=np.where(np.all(embed==x[0,:,i,j],axis=1))[0][0]
 #return encoding          