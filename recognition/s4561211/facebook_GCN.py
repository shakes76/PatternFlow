# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 21:31:12 2021

@author: Ya-Yu Kang
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
from random import sample
import matplotlib.pyplot as plt

from plot_TSNE import plot

def load_data(path):
    """
        Load data data and transform adjacency matrix
          
        Parameters:
        path (str): the path of data
        
        Returns:
        (scipy.sparse.coo.coo_matrix): Return adjacency matrix
        (scipy.sparse.csr.csr_matrix): Return features
        (torch.Tensor): Return labels
        
    """ 
    data = np.load(path)
    edges = data['edges']
    edges = np.unique(edges, axis=0)
    features = data['features']
    labels = data['target']
      
    features = sp.csr_matrix(features)
    A = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]))#.toarray()
      
    #normalize
    colsum = np.array(A.sum(0))
    D = np.power(colsum, -1)[0]
    D[np.isinf(D)] = 0
    D_inv = sp.diags(D)
    A_trans = D_inv.dot(A)
    
    #transform data type
    indices = torch.LongTensor(np.vstack((A_trans.tocoo().row, A_trans.tocoo().col)))
    values = torch.FloatTensor(A_trans.data)
    shape = A_trans.shape
      
    A_trans = torch.sparse_coo_tensor(indices, values, shape)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
       
    return A_trans, features, labels


class GCN(nn.Module):
    
    def __init__(self, n_feature, n_hidden, n_class, dropout):
        super(GCN, self).__init__()

        self.gc_layer1 = GraphConvolution(n_feature, n_hidden)
        self.gc_layer2 = GraphConvolution(n_hidden, n_class)
        self.dropout = dropout

    def forward(self, x, A):
        x = F.relu(self.gc_layer1(x, A))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc_layer2(x, A)
        
        return F.log_softmax(x, dim=1)
    

class GraphConvolution(Module):
    
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.W = torch.ones((in_features, out_features), requires_grad = True)
    
    def forward(self, input, A):
        weighted_feature = torch.mm(input, self.W)
        output = torch.mm(A, weighted_feature)
        
        return output
    
def data_index(tra_ratio,val_ratio):
    """
        Split the data into index
        
        Parameters:
        tra_ratio (float): ratio of training data
        val_ratio (float): ratio of validation data
    
        Returns:
        (list): the list of index for training data
        (list): the list of index for validation data
        (list): the list of index for test data
        
    """   
    sequence = [i for i in range(labels.shape[0])]
    tra = sample(sequence, int(labels.shape[0]*tra_ratio))
    rem = list(set(sequence).difference(set(tra)))
    val = sample(rem, int(labels.shape[0]*val_ratio))
    test = list(set(rem).difference(set(val)))
    return tra, val, test
  
def accuracy(output, labels):
    """
        Calculate accuracy 

        Parameters:
        output (Tensor): the log probability for each class
        labels (Tensor): the true labels
    
        Returns:
        (Tensor) : the accuracy 

    """
    pred = output.argmax(1)
    acc_ = torch.div(pred.eq(labels).sum(), labels.shape[0])
    return acc_

def loss(output,labels):
    """
        Calculate loss

        Parameters:
        output (Tensor): the log probability for each class
        labels (Tensor): the true labels
    
        Returns:
        (Tensor) : the loss

    """
    prab = output.gather(1, labels.view(-1,1))
    loss = -torch.mean(prab)
    return loss

def train_model(n_epochs):
    """
        Train the model and save the model with the largest accuracy for validation data

        Parameters:
        n_epochs (int): the number of iteration
        
        Returns:
        (list) : the accuracy for training data
        (list) : the accuracy for validation data
        
    """
    acc_pre = 0
    acc_tra = []
    acc_val = []
    for epoch in range(n_epochs):
        #train
        model.train() 
        optimizer.zero_grad()
        output = model(features, A)
        loss_ = loss(output[tra], labels[tra])
        accuracy_ = accuracy(output[tra], labels[tra])
        loss_.backward()
        optimizer.step()
        print('Epoch:',epoch,)
        print('train - loss:',loss_, ', accuracy', accuracy_)
        acc_tra += accuracy_,
        
        #validation
        loss_val = loss(output[val], labels[val])
        accuracy_val = accuracy(output[val], labels[val])
        print('validation - loss:',loss_val,', accuracy', accuracy_val)
        print('--------------------------------------------------')
        acc_val += accuracy_val,
        
        if acc_pre < accuracy_val:
            #save model
            torch.save(model.state_dict(), 'train_model.pth')
            acc_pre = accuracy_val
        
    return acc_tra, acc_val
        
def test_model():
    """
        Train the model and save the model with the largest accuracy for validation data

        Parameters:
        n_epochs (int): the number of iteration
        
    """
    model.load_state_dict(torch.load('train_model.pth'))
    output = model(features, A)
    loss_ = loss(output[test], labels[test])
    accuracy_ = accuracy(output[test], labels[test])
    print('test - loss:',loss_,', accuracy', accuracy_)
    

if __name__ == '__main__':
    #load data
    A, features, labels = load_data('facebook.npz')
    #split data index 
    #tra: val: test /0.2: 0.2: 0.6
    tra, val, test = data_index(0.2,0.2)
    #size of hidden layer
    hidden = 32

    model = GCN(n_feature=features.shape[1], n_hidden=hidden,
                n_class=len(np.unique(labels)), dropout=0.5)
    
    optimizer = optim.Adam([model.gc_layer1.W,model.gc_layer2.W], lr=0.01)
    
    acc_tra, acc_val = train_model(200)
    test_model()
    
    #plot accuracy
    plt.plot(range(200), acc_tra, 'b')
    plt.plot(range(200), acc_val, 'r')
    plt.legend(['accuracy_train', 'accuracy_validation'])
    plt.show()
    
    #plot tsne
    plot(labels, outputs = model.gc_layer1(features, A).detach().numpy())
    
    
    
    


