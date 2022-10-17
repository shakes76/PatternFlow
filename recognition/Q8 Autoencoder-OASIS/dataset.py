def DataProcess():
 """Function used to load the data, perform data augmentation, and create a dataloader

     Inputs:None

     Outputs:
       TRAINDATA: This is the augmented (includes images that are flipped upside down) Training dataset
       VALIDDATA: This is the validation data set
       TESTDATA:  This is the TESTDATA
       DATALOADER: This is the data loader constructed over the training dataset
 """   
 import numpy as np
 import torch
 import torchvision.transforms as transforms
 from PIL import Image
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
    img = Image.open(f'D:/Downloads/keras_png_slices_data/keras_png_slices_data/keras_png_slices_train/case_{h}_slice_{z}.nii.png').convert('RGB') #loads the image
    img=np.array(img) #convert the image into an array
    #to_tensor = transforms.ToTensor()
    #tensor = to_tensor(img)
    tensor=torch.tensor(img)/255 #converting the array so that pixels assume values between 0 and 1
    TRAINDATA[i,:,:,:]=tensor.float()
    TRAINDATA[9664+i,:,:,:]=torch.flipud(tensor).float() #adding the flipped version of the image to the dataset
 
    
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
    img = Image.open(f'D:/Downloads/keras_png_slices_data/keras_png_slices_data/keras_png_slices_validate/case_{h}_slice_{z}.nii.png').convert('RGB') #loads the image
    img=np.array(img) #convert the image into an array
   
  
    tensor=torch.tensor(img)/255 #converting the array so that pixels assume values between 0 and 1
    VALIDDATA[i,:,:,:]=tensor 

    import torchvision.transforms as transforms
 from PIL import Image
 TESTDATA=np.empty((544,256,256,3),dtype='float16') #initializing array to hold Testdata.It is in Float 16 due to memory issues.
 TESTDATA=torch.tensor(TESTDATA)
 for i in range(0,544): #This loop ranges across the test data and loads the images. An index j is used to record the number corresponding to the current block of 32 images. 
    j=int(np.floor(i/32)+441) #This sets the block number 
          

   
    
    h=f'{j}'       #h here is presenting the number j, corresponding to  which numbered block of 32 images is being dealt with.
    z=int(i-32*np.floor(i/32)) # z determines the current image number within the current block of 32 images
    img = Image.open(f'D:/Downloads/keras_png_slices_data/keras_png_slices_data/keras_png_slices_test/case_{h}_slice_{z}.nii.png').convert('RGB') #loads the image
    img=np.array(img) #convert the image into an array
    
    tensor=torch.tensor(img)/255 #converting the array so that pixels assume values between 0 and 1
    TESTDATA[i,:,:,:]=tensor
 return TRAINDATA,VALIDDATA,TESTDATA   