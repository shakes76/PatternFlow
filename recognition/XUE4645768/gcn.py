#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
import sys


# In[20]:


from sklearn.manifold import TSNE


# In[2]:


from sklearn import preprocessing  


# In[3]:


df=np.load('facebook.npz')
edges=df['edges']
features=df['features']
target=df['target']


# In[4]:


def normalize_adj(adjacency):
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()


# In[5]:


features=preprocessing.normalize(features)


# In[6]:


adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)


# In[7]:


adj = normalize_adj(adj)


# In[8]:


features = torch.FloatTensor(features)
labels = torch.LongTensor(target)


# In[9]:


id1=range(4494)
id2=range(4494,8988)
id3=range(8988,22470)


# In[10]:


num_nodes = features.shape[0]
train_mask = np.zeros(num_nodes, dtype=np.bool)
val_mask = np.zeros(num_nodes, dtype=np.bool)
test_mask = np.zeros(num_nodes, dtype=np.bool)


# In[11]:


train_mask[id1] = True
val_mask[id2] = True
test_mask[id3] = True


# In[12]:


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        "GC：L*X*\theta Args:input_dim: int output_dim: int"
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """Sparse matrix multiplication is used in calculation with A
    
        Args: 
        -------
            adjacency: torch.sparse.FloatTensor
            input_feature: torch.Tensor        
        """ 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        support = torch.mm(input_feature, self.weight.to(device))
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias.to(device)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# In[13]:


class GcnNet(nn.Module):
    """
    Define a model that contains two layers of gcn
    """
    def __init__(self, input_dim=128):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 32)
        self.gcn2 = GraphConvolution(32,8)
        
    
    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits=self.gcn2(adjacency, h)
        return logits


# In[14]:


# parameters
learning_rate = 0.01
weight_decay = 0.0005
epochs = 200


# In[15]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[16]:


model = GcnNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay) 

tensor_x = features.to(device)
tensor_y = labels.to(device)

tensor_train_mask = torch.from_numpy(train_mask).to(device)
tensor_val_mask = torch.from_numpy(val_mask).to(device)
tensor_test_mask = torch.from_numpy(test_mask).to(device)
indices = torch.from_numpy(np.asarray([adj.row, adj.col]).astype('int64')).long()
values = torch.from_numpy(adj.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices, values, (22470, 22470)).to(device)


# In[17]:


def train():
    
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(epochs):
        logits = model(tensor_adjacency, tensor_x)  # Forward propagation
        train_mask_logits = logits[tensor_train_mask]   # Only training selected nodes for supervising
        loss = criterion(train_mask_logits, train_y)    # loss
        optimizer.zero_grad()
        loss.backward()     # backward
        optimizer.step()    # update
        train_acc, _, _ = test(tensor_train_mask)     # train accuracy
        val_acc, _, _ = test(tensor_val_mask)      # val accuracy
        
        
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))
    
    


# In[18]:


def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()


# In[21]:


if __name__ == "__main__":
    print("Train model:")
    train()
    print("===============================================")
    test_accuracy, _, _ = test(tensor_test_mask)
    print("The test accuracy is：{:.4f}".format(test_accuracy))
    print("===============================================")
    print("Use TSNE to lower dimension")
    print("===============================================")
    test_accuracy, test_data, test_labels = test(tensor_test_mask)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500) # TSNE lowers dimension to 2
    low_dim_embs = tsne.fit_transform(test_data)
    plt.title('tsne result')
    plt.scatter(low_dim_embs[:,0], low_dim_embs[:,1], marker='o', c=test_labels)
    plt.savefig("tsne 3.png")


# In[ ]:




