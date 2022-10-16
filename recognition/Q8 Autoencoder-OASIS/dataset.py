def DataProcess():
 import numpy as np
 import torch
 import torchvision.transforms as transforms
 from PIL import Image
 TRAINDATA=np.empty((9664*2,256,256,3),dtype='float16')
 TRAINDATA=torch.tensor(TRAINDATA)
 for i in range(0,9664):
    j=int(np.floor(i/32)+1)
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

                            


    h=f'{j}'
    if j<10:
        h=f'00{j}'
    if j<100 and j>=10:
        h=f'0{j}'    
    z=int(i-32*np.floor(i/32))
    img = Image.open(f'D:/Downloads/keras_png_slices_data/keras_png_slices_data/keras_png_slices_train/case_{h}_slice_{z}.nii.png').convert('RGB')
    img=np.array(img)
    #to_tensor = transforms.ToTensor()
    #tensor = to_tensor(img)
    tensor=torch.tensor(img)/255
    TRAINDATA[i,:,:,:]=tensor.float()
 for i in range(0,9664):
    j=int(np.floor(i/32)+1)
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

                            


    h=f'{j}'
    if j<10:
        h=f'00{j}'
    if j<100 and j>=10:
        h=f'0{j}'    
    z=int(i-32*np.floor(i/32))
    img = Image.open(f'D:/Downloads/keras_png_slices_data/keras_png_slices_data/keras_png_slices_train/case_{h}_slice_{z}.nii.png').convert('RGB')
    img=np.array(img)
    #to_tensor = transforms.ToTensor()
    #tensor = to_tensor(img)
    tensor=torch.flipud(torch.tensor(img))/255
    TRAINDATA[9664+i,:,:,:]=tensor.float()
    
    import torchvision.transforms as transforms
 from PIL import Image
 VALIDDATA=np.empty((1120,256,256,3),dtype='float16')
 VALIDDATA=torch.tensor(VALIDDATA)
 for i in range(0,1120):
    j=int(np.floor(i/32)+402)
    #print(j)
    if j>=412:
        j=j+1
    if j>=414:
        j=j+1
    if j>=427:
        j=j+1
    if j>=436:
        j=j+1
    #if j>=73:
        #j=j+8
    #if j>=89:
        #j=j+1
    #if j>=92:
        #j=j+59         

    #if j<10:
       # h=f'00{j}'
    #if j<100 and j>10:
    h=f'{j}'    #h=f'0{j}'    
    z=int(i-32*np.floor(i/32))
    img = Image.open(f'D:/Downloads/keras_png_slices_data/keras_png_slices_data/keras_png_slices_validate/case_{h}_slice_{z}.nii.png').convert('RGB')
    img=np.array(img)
    #to_tensor = transforms.ToTensor()
    #tensor = to_tensor(img)
    tensor=torch.tensor(img)/255
    VALIDDATA[i,:,:,:]=tensor

    import torchvision.transforms as transforms
 from PIL import Image
 TESTDATA=np.empty((544,256,256,3),dtype='float16')
 TESTDATA=torch.tensor(TESTDATA)
 for i in range(0,544):
    j=int(np.floor(i/32)+441)
    #print(j)
    #if j>=412:
        #j=j+1
    #if j>=414:
        #j=j+1
    #if j>=427:
        #j=j+1
    #if j>=436:
        #j=j+1
    #if j>=73:
        #j=j+8
    #if j>=89:
        #j=j+1
    #if j>=92:
        #j=j+59         

    #if j<10:
       # h=f'00{j}'
    #if j<100 and j>10:
    h=f'{j}'    #h=f'0{j}'    
    z=int(i-32*np.floor(i/32))
    img = Image.open(f'D:/Downloads/keras_png_slices_data/keras_png_slices_data/keras_png_slices_test/case_{h}_slice_{z}.nii.png').convert('RGB')
    img=np.array(img)
    #to_tensor = transforms.ToTensor()
    #tensor = to_tensor(img)
    tensor=torch.tensor(img)/255
    TESTDATA[i,:,:,:]=tensor
 return TRAINDATA,VALIDDATA,TESTDATA   