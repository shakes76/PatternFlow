def VQVAE1(TRAINDATA,dimlatent,noembeddings,learningrate,commitcost):
 """This function trains VQVAE model on the given dataset with the chosen hyperparamters

     Input:
         TRAINDATA (torch tensor_): The training dataset over which the model is trained 
         DATALOADER               : The Dataloader constructed over the training dataset which is used to train the model
         dimlatent (float):       The dimensionality of the latent space that the output of the encoder is transformed into within the VQVAE layer
         noembeddings (float):    The number of embedding vectors which are possible latent space values for each sample
         learningrate (float_):   The learning rate of the VQVAE
         commitcost (float):      The weight in the loss function given to the Mean Squared Error associated with input:Latent Space representation and target: stop gradient function of the encoder output 

     Returns:
         _type_: _description_
  """
 from torch.utils.data import DataLoader
 import numpy as np
 import torch
 import torch.nn as nn
 import torch.nn.functional as F
 from torch.utils.data.sampler import SubsetRandomSampler
 from torch import flatten
 from torch.nn import ReLU

 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Setting the system so that the device used is GPU
 num_workers =0
 
 TRAINDATA=TRAINDATA.float()

 batch_size = 25 #setting Batch size
 num_train = len(TRAINDATA)
 indices = list(range(num_train))

 np.random.shuffle(indices) #randomly shuffling the data

 train_index= indices

 train_sampler = SubsetRandomSampler(train_index) #creating sampler for the training data

 train_loader = torch.utils.data.DataLoader(TRAINDATA, batch_size = batch_size,
                                           sampler = train_sampler, num_workers = num_workers) #setting up the data load



 class indeed(nn.Module):  #Model is created as class 'indeed'
   def __init__(self, numembedding, embeddingdim,commitcost): #initializing the class
        
        super(indeed, self).__init__()
        
        self.numembedding=numembedding #initializing the number of embeddings, the dimensionality of the embeddings and the commitment cost which is used in the loss function
        self.embeddingdim=embeddingdim
        self.commitcost=commitcost
        #Initializing the neural network layers, in this case I created four convolutional neural network layers. I also used Batch Normalization
        self.layer0=nn.Conv2d(3,int(np.round(self.embeddingdim/2)),kernel_size=3,stride=1,padding='same')
        self.layer1=nn.BatchNorm2d(int(np.round(self.embeddingdim/2)))
        self.layer2=nn.Conv2d(int(np.round(self.embeddingdim/2)),int(self.embeddingdim),kernel_size=3,stride=1,padding='same')
        self.layer3=nn.BatchNorm2d(int(self.embeddingdim))
        #The four lines above correspond to the encoder
        
        self.layer4=nn.Conv2d(int(self.embeddingdim),int(np.round(self.embeddingdim/2)),kernel_size=3,stride=1,padding='same')
        self.layer5=nn.BatchNorm2d(int(np.round(self.embeddingdim/2)))
        self.layer6=nn.Conv2d(int(np.round(self.embeddingdim/2)),3,kernel_size=3,stride=1,padding='same')
        #The three lines above correspond to the decoder
        
        
        
   def VQVAE(self,x,numembedding,embeddingdim,commitcost):#Vector Quantization Layer
        x = x.permute(0, 2, 3, 1).contiguous()  #Reshaping the output of the encoder dataset to be "number of images*height*width*channels"
        embedding=nn.Embedding(numembedding,embeddingdim)   #setting up the embeddings, where the values in the embedding are sampled from a uniform distribution
        embedding.weight.data.uniform_(-2/numembedding, 2/numembedding)
        embedding.to(device)
        #rearrange x?
        flat_x = x.reshape(-1, embeddingdim) #flattening the array so the number of columns correspond to the number of embedding dimensions
        dist=(torch.sum(flat_x**2,dim=1,keepdim=True)+torch.sum(embedding.weight**2,dim=1)-2*torch.matmul(flat_x,embedding.weight.t())).to(device)
        #The line above calculates the euclidean norm squared with each sample of the flattened output of the encoder with each embedding vector. Wh
        
        indexes=torch.argmin(dist,dim=1).unsqueeze(1) #This determines which embedding vector minimizes euclidean norm between it and each sample of the flattened encoder output
        coded = torch.zeros(indexes.shape[0], self.numembedding, device=x.device)
        coded.scatter_(1,indexes, 1)
        quant=torch.mm(coded, embedding.weight).reshape(x.shape)# Array whose rows consist of the minimizing embedding vector for the corresponding sample.This is the latent space representation 

        #quant=embedding.weight[indexes,:].reshape(x.shape)
        loss = F.mse_loss(quant.detach(), x)+commitcost* F.mse_loss(quant, x.detach()) #non-reconstruction loss part of the loss function
        
        
        quant = x + (quant - x).detach() #straight through estimator of the minimizing embedding vector/latent space representation
        return loss, quant.permute(0, 3, 1, 2).contiguous()


   #def sampling(self,mu,logvariance):
      

   
        #self.fc2 = nn.Linear(in_features=100, out_features=10)
   def forward(self, x):
        
        #This a function which implements the layers
        x=torch.nn.functional.relu(self.layer0(x))
        x=self.layer1(x)
        x=torch.nn.functional.relu(self.layer2(x))
        x=self.layer3(x)
        #The above code corresponds to the encoder
        Loss,z=self.VQVAE(x,self.numembedding,self.embeddingdim,self.commitcost) #vector quantization/latent space determining layer
        #The remaining three lines of this function correspond to the decoder
        x=torch.nn.functional.relu(self.layer4(z))
        x=self.layer5(x)
        x=torch.nn.functional.sigmoid(self.layer6(x))
        return x,Loss
 model = indeed(numembedding=noembeddings,embeddingdim=dimlatent,commitcost=commitcost)
 model.to(device)
 EPOCHS=30 #number of epochs

 opt = torch.optim.Adam(model.parameters(), lr=learningrate,weight_decay=0.0005) #setting up the optimizer
 lossFn1 = nn.BCELoss(reduction='mean') #setting up the reconstruction loss which is the mean Binary Cross-Entroy


 E=np.empty((EPOCHS))
 E1=np.empty((EPOCHS))
 for e in range(0, EPOCHS): #iterating through the epochs and training the model
  print(e)
  acc=0  
  model.train() #setting the mode to train
    
  totalbinaryentropyloss = 0 #setting the initial values of the reconstruction loss and the non-reconstruction part to 0
  totalnonreconloss=0
 
  i=0   # loop over the training set
  for x in train_loader: #looping through the batches
        
        pred,Loss1 = model(x.float().to(device)) #The model is run, and predictions and non-reconstruction loss for the batch are computed
        Loss2=lossFn1(pred.to(device),x.float().to(device)) #calculate reconstruction loss
        loss = Loss1+Loss2 #Calculate total loss as sum of reconstruction and non-reconstruction loss component
        
        opt.zero_grad()   #This line and the next two lines calculating the gradients and updating the weights
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalbinaryentropyloss += len(x)*Loss2*256*256*3 #computing the sum of the reconstruction loss over the batch and adding it to reconstruction loss sum of the previous batches
        totalnonreconloss+=len(x)*Loss1*256*256*dimlatent#computing the sum of the reconstruction loss over the batch and adding it to reconstruction loss sum of the previous batches
        #m=nn.Softmax(dim=1)
        #acc=acc+np.sum(np.array(torch.argmax(m(pred.cpu()),dim=1))==torch.argmax(x[:,3072:3082],dim=1).detach().numpy())/len(x)

   
   
 #ACC[e]=acc/i
  E1[e]=totalbinaryentropyloss/(len(TRAINDATA)*256*256*3) #Computing the mean total binary cross-entropyloss
  E[e]=totalnonreconloss/(len(TRAINDATA)*256*256*dimlatent)+totalbinaryentropyloss/(len(TRAINDATA)*256*256*3) # Computing the mean of the total loss

 #The remaining code plots the training loss versus the cpoch number

 import matplotlib.pyplot as plt
 from matplotlib.pyplot import figure

 ep=range(1,EPOCHS+1)

 plt.plot(ep,E)
 plt.title(f'Average Loss(with Binary Cross-Entropy Reconstruction Loss component) Loss Versus Epoch No for {dimlatent} Dimension Latent Space, {noembeddings} Embeddings, Learning Rate of {learningrate} and {commitcost} Commitment Cost')
 plt.xlabel('Epoch No')
 plt.ylabel('Average  Loss') 
 plt.show()
 plt.plot(ep,E1)
 plt.title(f'Average Binary Cross-Entropy Reconstruction Loss Versus Epoch No for {dimlatent} Dimension Latent Space, {noembeddings} Embeddings, Learning Rate of {learningrate} and {commitcost} Commitment Cost')
 plt.xlabel('Epoch No')
 plt.ylabel('Average Binary Cross-Entropy Loss') 
 plt.show()
 
 return model

