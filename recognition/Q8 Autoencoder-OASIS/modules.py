def VQVAE1(TRAINDATA,dimlatent,noembeddings,learningrate,commitcost):

 from torch.utils.data import DataLoader
 import numpy as np
 import torch
 import torch.nn as nn
 import torch.nn.functional as F
 from torch.utils.data.sampler import SubsetRandomSampler
 from torch import flatten
 from torch.nn import ReLU

 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 num_workers =8
   #train_data=torch.Tensor(x_tr)
   #x_train=train_data.reshape(len(train_data), -1).float()
   #y_train=torch.Tensor(y_tr)
#train_data=np.concatenate((X_train,y_train),axis=1)
 TRAINDATA=TRAINDATA.float()
#test_data=torch.tensor(x_test)#Y=(data[:,3])
#y_test=torch.tensor(y_test)
#test_data=np.concatenate((x_test,y_test),axis=1)
 batch_size = 50
 num_train = len(TRAINDATA)
 indices = list(range(num_train))

 np.random.shuffle(indices)

 train_index= indices

 train_sampler = SubsetRandomSampler(train_index)

 train_loader = torch.utils.data.DataLoader(TRAINDATA, batch_size = batch_size,
                                           sampler = train_sampler, num_workers = num_workers)

#dataloader = DataLoader(DatasetWrapper(x), batch_size=batch_size, shuffle=False)

 class indeed(nn.Module):
   def __init__(self, numembedding, embeddingdim,commitcost):
        # call the parent constructor
        super(indeed, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.numembedding=numembedding
        self.embeddingdim=embeddingdim
        self.commitcost=commitcost
        self.layer0=nn.Conv2d(3,embeddingdim/2,kernel_size=3,stride=1,padding='same')
        self.layer1=nn.BatchNorm2d(embeddingdim/2)
        self.layer2=nn.Conv2d(embeddingdim/2,embeddingdim,kernel_size=3,stride=1,padding='same')
        self.layer3=nn.BatchNorm2d(embeddingdim)
        self.layer4=nn.Conv2d(embeddingdim,embeddingdim/2,kernel_size=3,stride=1,padding='same')
        self.layer5=nn.BatchNorm2d(embeddingdim/2)
        self.layer6=nn.Conv2d(embeddingdim/2,3,kernel_size=3,stride=1,padding='same')
        
        
        
   def VQVAE(self,x,numembedding,embeddingdim,commitcost):
        x = x.permute(0, 2, 3, 1).contiguous()
        embedding=nn.Embedding(numembedding,embeddingdim)
        embedding.weight.data.uniform_(-2/numembedding, 2/numembedding)
        embedding.to(device)
        #rearrange x?
        flat_x = x.reshape(-1, embeddingdim)
        dist=(torch.sum(flat_x**2,dim=1,keepdim=True)+torch.sum(embedding.weight**2,dim=1)-2*torch.matmul(flat_x,embedding.weight.t())).to(device)
        
        #dist=torch.tensor(np.empty((flat_x.shape[0],embedding.weight.shape[0])))
        #for i in range(0,numembedding):
         #print(flat_x.is_cuda)
         #print(embedding.weight.is_cuda)
         #dist[:,i]=(torch.linalg.vector_norm(flat_x.float()-embedding.weight[i,:]*torch.ones(flat_x.shape[0],flat_x.shape[1]).float().to(device),ord=2,dim=1).to(device))**2
        indexes=torch.argmin(dist,dim=1).unsqueeze(1)
        coded = torch.zeros(indexes.shape[0], self.numembedding, device=x.device)
        coded.scatter_(1,indexes, 1)
        quant=torch.mm(coded, embedding.weight).reshape(x.shape)

        #quant=embedding.weight[indexes,:].reshape(x.shape)
        loss = F.mse_loss(quant.detach(), x)+commitcost* F.mse_loss(quant, x.detach())
        
        
        quant = x + (quant - x).detach()
        return loss, quant.permute(0, 3, 1, 2).contiguous()


   #def sampling(self,mu,logvariance):
      

   
        #self.fc2 = nn.Linear(in_features=100, out_features=10)
   def forward(self, x):
        
        
        x=torch.nn.functional.relu(self.layer0(x))
        x=self.layer1(x)
        x=torch.nn.functional.relu(self.layer2(x))
        x=self.layer3(x)
        #print(self.numembedding)
        #print(self.embeddingdim)
        #print(self.commitcost)
        #x.self.numembedding,self.embeddingdim,self.commitcost
        Loss,z=self.VQVAE(x,self.numembedding,self.embeddingdim,self.commitcost)
        x=torch.nn.functional.relu(self.layer4(z))
        x=self.layer5(x)
        x=torch.nn.functional.sigmoid(self.layer6(x))
        return x,Loss
 model = indeed(numembedding=noembeddings,embeddingdim=dimlatent,commitcost=commitcost)
 model.to(device)
 EPOCHS=30

 opt = torch.optim.Adam(model.parameters(), lr=learningrate,weight_decay=0.0005)
 lossFn1 = nn.BCELoss(reduction='mean')


 E=np.empty((EPOCHS))
 for e in range(0, EPOCHS):
  print(e)
  acc=0  
  model.train()
    
  totalbinaryentropyloss = 0
 
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
        totalnonreconloss+=len(x)*Loss1*256*256*dimlatent
        #m=nn.Softmax(dim=1)
        #acc=acc+np.sum(np.array(torch.argmax(m(pred.cpu()),dim=1))==torch.argmax(x[:,3072:3082],dim=1).detach().numpy())/len(x)

   
   
 #ACC[e]=acc/i
  E[e]=totalnonreconloss/(len(TRAINDATA*256*256*dimlatent))+totalbinaryentropyloss/(len(TRAINDATA)*256*256*3)
 import matplotlib.pyplot as plt
 from matplotlib.pyplot import figure

 ep=range(1,EPOCHS+1)
#figure, axis = plt.subplots(1, 1)
#axis[0].plot(ep,ACC)
#axis[0].set_title('Accuracy Versus Epoch No')
#axis[0].set_xlabel('Epoch No')
#axis[1].set_ylabel('Accuracy')
 plt.plot(ep,E)
 plt.title(f'Average Loss(with Binary Crossentropy reconstruction loss component) Loss Versus Epoch No for {dimlatent} Dimension Latent Space,{noembeddings} Embeddings, Learning Rate of {learningrate} and {commitcost} Commitment Cost')
 plt.xlabel('Epoch No')
 plt.ylabel('Average Binary Cross-Entropy Loss') 
 plt.show()
   #correct = np.array(out.cpu())==np.array(y_val.reshape((10000)))
   

        #trainCorrect += (pred.argmax(1) == ytarget).type(
#m=nn.Softmax(dim=1)
 return model

