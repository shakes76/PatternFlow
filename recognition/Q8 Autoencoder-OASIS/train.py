
def TRAINVALTEST():
 import modules
 from modules import VQVAE1
 from dataset import DataProcess
 from pandas import DataFrame
 import numpy as np
 from torchmetrics import StructuralSimilarityIndexMeasure
 import torch   
 TRAINDATA,VALIDDATA,TESTDATA=DataProcess()
 TRAINDATA=TRAINDATA.permute(0, 3, 2, 1).contiguous()
 VALIDDATA=VALIDDATA.permute(0, 3, 2, 1).contiguous()
 TESTDATA=TESTDATA.permute(0,3,2,1).contiguous()
 DimLatSpace=np.array([10,20,40])
 Numberofembeddings=np.array([10,30,100])
 learningrate=np.array([0.00002,0.0001,0.001])
 commitcost=np.array([0.1,0.3,0.6])
 mod=DataFrame(np.empty((3,3,3,3)))
 predict=np.empty((len(VALIDDATA),VALIDDATA.shape[1],VALIDDATA.shape[2],VALIDDATA.shape[3]))
 metric =StructuralSimilarityIndexMeasure(data_range=1.0,reduction='sum')
 ssim=np.empty((3,3,3,3))
 for i in range(0,2):
    for j in range(0,2):
        for z in range(0,2):
            for k in range(0,2):
             mod.iloc[i,j,z,k]=VQVAE1(TRAINDATA,DimLatSpace[i],Numberofembeddings[j],learningrate[z],commitcost[k])
             
             for t in range(0,len(VALIDDATA)):
              predict[t,:,:,:]=mod.iloc[i,j,z,k](VALIDDATA[t].cuda().float().reshape((1,VALIDDATA[t].shape[0],VALIDDATA[t].shape[1],VALIDDATA[t].shape[2])))[0].cpu().detach().numpy()
              #Predictions were made, one at a time, according to the line above due to memory issues that result if you try make predictions for the entire Validation dataset all at once
              ssim=0
              for g in range(0,len(VALIDDATA/10)):
               ssim[i,j,z,k]=metric(torch.tensor(np.float32(predict[range(i*10,(i+1)*10),:])),VALIDDATA.to(torch.float32)[range(i*10,(i+1)*10),:])+ssim
               ssim[i,j,z,k]=ssim[i,j,z,k]/len(VALIDDATA) #issues arise trying to compute SSIM for entire dataset all at once so the SSIM sums are calculated piecemeal
 maxssim=max(ssim)
 indices=np.where(ssim==maxssim)
 predict=mod.iloc[indices[0],indices[1],indices[2],indices[3]]((TESTDATA.cuda().float().reshape((1,TESTDATA[t].shape[0],TESTDATA[t].shape[1],TESTDATA[t].shape[2])))[0].cpu().detach().numpy())
 ssimtest=0
 for i in range(0,int(len(TESTDATA)/4)):
  ssimtest=metric(torch.tensor(np.float32(predict[range(i*4,(i+1)*4),:])),TESTDATA.to(torch.float32)[range(i*4,(i+1)*4),:])+ssim
 ssimtest=ssimtest/len(TESTDATA)
 import dill as pickle
 with open('goodmodel2.pkl', 'wb') as file:
  pickle.dump(mod.iloc[indices[0],indices[1],indices[2],indices[3]], file)
 return mod,ssim,ssimtest             