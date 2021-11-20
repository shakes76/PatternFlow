import random
import math
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.manifold import TSNE

class GCNLayer(Module):
    def __init__(self, n_in_features, n_out_features):
        super(GCNLayer, self).__init__()
        self.n_in_features = n_in_features
        self.n_out_features = n_out_features
        self.weights = Parameter(torch.ones([n_in_features, n_out_features], dtype=torch.float32))

    def forward(self, input, adj):
        linear = torch.mm(input, self.weights) #WX
        output =  torch.mm(adj, linear) #D^-1*A*WX

        return output

class GCNModel(nn.Module):
    def __init__(self, n_class, n_in_features, drop_p=0.5):
        super(GCNModel, self).__init__()

        self.gcn1 = GCNLayer(n_in_features, 64)
        self.gcn2 = GCNLayer(64, n_class)
        self.drop_p = drop_p

    def forward(self, input:torch.FloatTensor, adj:torch.FloatTensor, TSNE_collect=False):
        x1 = self.gcn1(input, adj)
        x1_relu = F.relu(x1)
        x1_drop = F.dropout(x1_relu, self.drop_p, training=self.training)
        x2 = self.gcn2(x1_drop, adj)
        
        log_softmax = nn.LogSoftmax(dim=1)

        if TSNE_collect:
            self.gcn1_embed = x1

        return log_softmax(x2)

class Facebook_Node_Classifier():
    def __init__(self, facebook_file: str, train_ratio=.6, val_ratio=0.2, random_state=25):
        self.seed = random_state
        self.data_process(facebook_file, train_ratio, val_ratio)
        self.create_adj()
        self.model = GCNModel(self.n_class, self.n_features)
        
    def data_process(self, facebook_file: str, train_ratio, val_ratio):
        data = np.load(facebook_file)

        edges = data['edges']
        features = data['features']
        target = data['target']

        self.n_edges = edges.shape[0]
        self.n_features = features.shape[1]
        self.n_node = target.shape[0]
        self.n_class = len(np.unique(target))

        self.node_features = torch.FloatTensor(features)
        self.target = torch.LongTensor(target)
        self.edges = edges

        ##Split traing val test index
        random.seed(self.seed)
        n_train = math.floor(self.n_node*train_ratio)
        n_val = math.floor(self.n_node*val_ratio)
        total_idx = list(range(self.n_node))
        train_idx = random.sample(total_idx, n_train)
        valtest = list(set(total_idx).difference(set(train_idx)))
        val_idx = random.sample(valtest, n_val)
        test_idx = list(set(valtest).difference(set(val_idx)))           
        
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        
    def create_adj(self):
        #create an iniitial adj matrix for sparse matrix
        adj = sp.coo_matrix((np.ones(self.n_edges), (self.edges[:, 0], self.edges[:, 1])))
        
        #make sure all element is 1 or 0
        adj_t = torch.Tensor(adj.toarray())
        adj_t[adj_t > 0] = 1

        #check adj is semetric or not
        assert sum(sum(adj_t != adj_t.T)) == 0, 'Adjacency matrix is not symetric'

        #normalise 
        rowsum = np.array(adj_t.sum(1))
        inv = np.ma.power(rowsum, -1) #creaet D^-1
        inv[inv == np.inf] = 0.#if 0 is inv 
        D_inv = sp.diags(inv)
        adj_m = D_inv.dot(adj_t) #D^-1*A

        self.adj = torch.FloatTensor(adj_m)

    def get_acc(self, output:torch.FloatTensor, target):
        prediction = output.argmax(1)
        correct = prediction == target
        
        return sum(correct)/len(target)
        
    def train_modle(self, n_epoch=30, lr=0.01):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.train_accs = []
        self.val_accs = []
        best_acc = 0
        
        for epoch in range(n_epoch):
            self.model.train()
            optimizer.zero_grad()
            
            output = self.model(self.node_features, self.adj)
            #Compute loss
            loss = nn.NLLLoss()
            loss_out = loss(output[self.train_idx], self.target[self.train_idx])
            #Back Propregation
            loss_out.backward()
            optimizer.step()

            acc = self.get_acc(output[self.train_idx], self.target[self.train_idx])
            self.train_accs.append(float(acc))
            print("Epoch: " + str(epoch) +" Accuracy: " + str(acc) + " Loss: " + str(loss_out))

            val_acc = self.get_acc(output[self.val_idx], self.target[self.val_idx])
            self.val_accs.append(float(val_acc))
            print("Val Acc:" + str(val_acc))

            if best_acc < val_acc:
                best_acc = val_acc

                torch.save(self.model.state_dict(), 'model_w.pth')
                print("Save model parameters----------------------------------------")

    def test_model(self):
        self.model.load_state_dict(torch.load('model_w.pth'))
        output = self.model(self.node_features, self.adj)
        loss = nn.NLLLoss()
        loss_out = loss(output[self.test_idx], self.target[self.test_idx])
        acc = self.get_acc(output[self.test_idx], self.target[self.test_idx])

        print("Test Performance: ")
        print("Accuracy: " + str(acc) + " Loss: " + str(loss_out))


    def plot_truth_TSNE(self):
        plt.figure(figsize=(15, 8))
        tsne = TSNE(random_state=self.seed).fit_transform(self.node_features)
        palette = sns.color_palette("muted", self.n_class)
        tsne_plot = sns.scatterplot(tsne[:,0], tsne[:,1], hue=self.target.numpy(), legend='full', palette=palette, s=10)
        fig = tsne_plot.get_figure()
        fig.savefig("tsne_original_plot")

    def plot_GCN_TSNE(self):
        plt.figure(figsize=(15, 8))
        self.model.load_state_dict(torch.load('model_w.pth'))
        _ = self.model(self.node_features, self.adj, TSNE_collect=True)

        embedding = self.model.gcn1_embed.detach().numpy()
        
        tsne = TSNE(random_state=self.seed).fit_transform(embedding)
        palette = sns.color_palette("muted", self.n_class)
        
        tsne_plot = sns.scatterplot(tsne[:,0], tsne[:,1], hue=self.target.numpy(), legend='full', palette=palette, s=10)
        fig = tsne_plot.get_figure()
        fig.savefig("tsne_embedding_plot") 

    def plot_accuracy(self):
        plt.figure(figsize=(15, 8))
        plt.plot(list(range(len(self.train_accs))), self.train_accs , label = "Training Acc")
        plt.plot(list(range(len(self.val_accs))), self.val_accs, label = "Validation Acc")
        plt.xlabel('Epoch')
        # Set the y axis label of the current axis.
        plt.ylabel('Accuracy')
        # Set a title of the current axes.
        plt.title('Accuracy trend')
        # show a legend on the plot
        plt.legend()
        plt.savefig("Accuracy_Trend.png")

if __name__ == "__main__":
    
    facebook_path = 'facebook.npz'

    classifer = Facebook_Node_Classifier(facebook_file=facebook_path, train_ratio=0.2, val_ratio=0.2)

    classifer.train_modle(n_epoch=200)

    classifer.test_model()

    classifer.plot_accuracy()

    classifer.plot_truth_TSNE()
    
    classifer.plot_GCN_TSNE()
    
