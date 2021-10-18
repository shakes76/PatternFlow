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
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
from random import sample

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
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]))#.toarray()
      
    #normalize
    colsum = np.array(adj.sum(0))
    D = np.power(colsum, -1)[0]
    D[np.isinf(D)] = 0
    D_inv = sp.diags(D)
    adj_trans = D_inv.dot(adj)
    
    #transform data type
    indices = torch.LongTensor(np.vstack((adj_trans.tocoo().row, adj_trans.tocoo().col)))
    values = torch.FloatTensor(adj_trans.data)
    shape = adj_trans.shape
      
    adj_trans = torch.sparse_coo_tensor(indices, values, shape)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
       
    return adj_trans, features, labels


class GCN(nn.Module):
    
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        
        return F.log_softmax(x, dim=1)
    

class GraphConvolution(Module):
    
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.W = torch.ones((in_features, out_features), requires_grad = True)
    
    def forward(self, input, adj):
        weighted_feature = torch.mm(input, self.W)
        output = torch.mm(adj, weighted_feature)
        
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
    pred = output.max(1)[1].type_as(labels)
    #= pred.eq(labels).sum()/ labels.shape[0]
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
        
    """
    acc_pre = 0
    
    for epoch in range(n_epochs):
        #train
        model.train() 
        optimizer.zero_grad()
        output = model(features, adj)
        loss_ = loss(output[tra], labels[tra])
        accuracy_ = accuracy(output[tra], labels[tra])
        loss_.backward()
        optimizer.step()
    
        #validation
        loss_val = loss(output[val], labels[val])
        accuracy_val = accuracy(output[val], labels[val])
        print('validation - Epoch:',epoch, ', loss:',loss_val,', accuracy', accuracy_val)
        
        if acc_pre < accuracy_val:
            #save model
            torch.save(model.state_dict(), 'train_model.pth')
    
        acc_pre = accuracy_val
        
def test_model():
    """
        Train the model and save the model with the largest accuracy for validation data

        Parameters:
        n_epochs (int): the number of iteration
        
    """
    model.load_state_dict(torch.load('train_model.pth'))
    output = model(features, adj)
    loss_ = loss(output[test], labels[test])
    accuracy_ = accuracy(output[test], labels[test])
    print('test - loss:',loss_,', accuracy', accuracy_)
    




