def VQVAE(TRAINDATA):

 from torch.utils.data import DataLoader
 import numpy as np
 import torch
 import torch.nn as nn
 from torch.utils.data.sampler import SubsetRandomSampler
 from torch import flatten
 from torch.nn import ReLU

 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 num_workers = 0
   #train_data=torch.Tensor(x_tr)
   #x_train=train_data.reshape(len(train_data), -1).float()
   #y_train=torch.Tensor(y_tr)
#train_data=np.concatenate((X_train,y_train),axis=1)
 TRAINDATA=TRAINDATA.float()
#test_data=torch.tensor(x_test)#Y=(data[:,3])
#y_test=torch.tensor(y_test)
#test_data=np.concatenate((x_test,y_test),axis=1)
 batch_size = 100
 num_train = len(TRAINDATA)
 indices = list(range(num_train))

 np.random.shuffle(indices)

 train_index= indices

 train_sampler = SubsetRandomSampler(train_index)

 train_loader = torch.utils.data.DataLoader(TRAINDATA, batch_size = batch_size,
                                           sampler = train_sampler, num_workers = num_workers)

#dataloader = DataLoader(DatasetWrapper(x), batch_size=batch_size, shuffle=False)

 class indeed(nn.Module):
   def __init__(self, numembedding, embeddingdim):
        # call the parent constructor
        super(indeed, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.numembedding=numembedding
        self.embeddingdim=embeddingdim
        self.layer0=nn.Conv2d(3,32,kernel=3,stride=1,padding='same')
        self.layer2=nn.BatchNorm2d(32)
        self.layer1=nn.Conv2d(3,64,kernel=3,stride=1,padding='same')
        self.layer2=nn.BatchNorm2d(64)
        self.layer3=nn.Conv2d(64,32,kernel=3,stride=1,padding='same')
        self.layer2=nn.BatchNorm2d(32)
        self.layer4=nn.Conv2d(32,3,kernel=3,stride=1,padding='same')
        
        
        
   def VAVAE(x,numembedding,embeddingdim,commitcost):
        x = x.permute(0, 2, 3, 1).contiguous()
        embedding=nn.Embedding(numembedding,embeddingdim)
        embedding.weight.data.uniform_(-1/numembedding, 1/numembedding)
        #rearrange x?
        flat_input = x.view(-1, embeddingdim)
        dist=torch.tensor(np.empty((x.shape[0],embedding.weight.shape[0])))
        for i in range(0,numembedding):
         dist[:,i]=torch.linalg.vector_norm(x.float(),embedding.weight[i,:]*torch.ones(x.shape[0],x.shape[1].float()))**2
        indexes=torch.argmin(dist,dim=1)
        quant=embedding.weight[indexes,:]
        loss1 = F.mse_loss(quant.detach(), x)
        loss2 =commitcost* F.mse_loss(quant, x.detach())
        Loss=loss1+loss2
        quant = x + (quant - x).detach()
        return Loss, quant.permute(0, 3, 1, 2).contiguous()


   #def sampling(self,mu,logvariance):
      

   
        #self.fc2 = nn.Linear(in_features=100, out_features=10)
   def forward(self, x,numembedding,embeddingdim,commitcost):
        
        
        x=torch.nn.functional.relu(self.layer(x))
        x=x.reshape((len(x),10,5,5))
        x=torch.nn.functional.relu(self.layer0(x))
        x=x.reshape((len(x),90))
        Loss,z=VAE(x,self.numembedding,self.embeddingdim,commitcost)
        #logvariance=self.layer2(x)
        
        #z=self.sampling(mu,logvariance)
        x=torch.nn.functional.relu(self.layer3(z))
        x=torch.sigmoid(self.layer5(torch.nn.functional.relu(self.layer4(x))))#x=self.relu0(x)
        
        
        
        return x,Loss
 model = indeed(numChannels=1,classes=10)
 model.to(device)
 EPOCHS=12

 opt = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0005)
 lossFn1 = nn.BCELoss(reduction='mean')


 E=np.empty((EPOCHS))
 for e in range(0, EPOCHS):
  print(e)
  acc=0  
  model.train()
    
  totalTrainLoss = 0
 
  i=0   # loop over the training set
  for x in train_loader:
        #i=i+1
        # send the input to the device
        #(x, y) = (x.to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        #l=len(x)
        pred,Loss1 = model(x.float().to(device))
        Loss2=lossFn1(pred.to(device),x.float().to(device))
        loss = Loss1+Loss2
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalbinaryentropyloss += len(x)*Loss2*256*256*3
        #m=nn.Softmax(dim=1)
        #acc=acc+np.sum(np.array(torch.argmax(m(pred.cpu()),dim=1))==torch.argmax(x[:,3072:3082],dim=1).detach().numpy())/len(x)

   
   
 #ACC[e]=acc/i
  E[e]=totalbinaryentropyloss/(len(TRAINDATA)*256*256*3)
 import matplotlib.pyplot as plt
 from matplotlib.pyplot import figure

 ep=range(1,EPOCHS+1)
#figure, axis = plt.subplots(1, 1)
#axis[0].plot(ep,ACC)
#axis[0].set_title('Accuracy Versus Epoch No')
#axis[0].set_xlabel('Epoch No')
#axis[1].set_ylabel('Accuracy')
 plt.plot(ep,E)
 plt.title('Average Binary Cross-Entropy Loss Versus Epoch No')
 plt.xlabel('Epoch No')
 plt.ylabel('Average Binary Cross-Entropy Loss')
 plt.show()
   #correct = np.array(out.cpu())==np.array(y_val.reshape((10000)))
   

        #trainCorrect += (pred.argmax(1) == ytarget).type(
#m=nn.Softmax(dim=1)
 return model

