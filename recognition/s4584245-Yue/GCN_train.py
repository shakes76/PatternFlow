# Imort the packages.
from load_data import *
from GCN_Layer import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#model Hyperparameter.
learning_rate = 0.1
weight_decay = 5e-4
epochs = 150

#Model definition:Model, Loss, Optimizer.
device = "cpu"
model = GCN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# Feature and labels.
tensor_x = features.to(device)
tensor_y = labels.to(device)
#Split the data.(load_data part)
tensor_train_mask = torch.from_numpy(train_mask).to(device)
tensor_val_mask = torch.from_numpy(val_mask).to(device)
tensor_test_mask = torch.from_numpy(test_mask).to(device)
indices = torch.from_numpy(np.asarray([adjacency.row, adjacency.col]).astype('int64')).long()
values = torch.from_numpy(adjacency.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices, values, (22470, 22470)).to(device)

class Train:

    def test(self,capsys):
        # Tset the GCN model.
        model.eval()
        with torch.no_grad():
            logits = model(tensor_adjacency, tensor_x)
            test_mask_logits = logits[capsys]
            predict_y = test_mask_logits.max(1)[1]
            accuarcy = torch.eq(predict_y, tensor_y[capsys]).float().mean()
        return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[capsys].cpu().numpy()

    def train(self):
        #Train the GCN model
        self.loss_history = []
        self.val_acc_history = []
        model.train()
        train_y = tensor_y[tensor_train_mask]
        for epoch in range(epochs):
            logits = model(tensor_adjacency, tensor_x)# Forward propagation
            train_mask_logits = logits[tensor_train_mask]
            loss = criterion(train_mask_logits, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc, _, _ = self.test(tensor_train_mask)
            val_acc, _, _ = self.test(tensor_val_mask)
            #Record the change of loss value and accuracy rate during training for drawing
            self.loss_history.append(loss.item())
            self.val_acc_history.append(val_acc.item())
            # Print GCN model result.
            print("Epoch %3d: Loss %.4f, Train_accuracy %.4f, Validation_accuracy %.4f" % (
                epoch, loss.item(), train_acc.item(), val_acc.item()))
        return self.loss_history, self.val_acc_history


    def acc(self):
        # Drawing the accuracy trend.
        plt.plot(self.val_acc_history, label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.2, 1])
        plt.legend(loc='lower right')
        plt.savefig("acc.png")
        plt.show()
