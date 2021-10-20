# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 21:10:24 2021

@author: Ya-Yu Kang
"""
from sklearn.manifold import TSNE
import seaborn as sns

def plot(labels, outputs ):
    """
        Plot TSNE embeddings 

        Parameters:
        A (scipy.sparse.coo.coo_matrix): adjacency matrix
        features (scipy.sparse.csr.csr_matrix): features
        labels (torch.Tensor): labels
        
    """
    
    tsne = TSNE().fit_transform(outputs)
    
    sns.set(rc={'figure.figsize':(11,8)})
    palette = sns.color_palette("muted", 4)
          
    sns.scatterplot(tsne[:,0], tsne[:,1], hue=labels, legend='full', palette=palette, s=10)