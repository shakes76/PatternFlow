import torch.optim as optim
import torch.nn.functional as func
import time
import pandas as pd
import numpy as np
import umap
import umap.plot

import modules
import dataset

# Load and process data
data = dataset.load_data()
train_ds, valid_ds, test_ds, features, target, edges = dataset.process_data(data)
adj = dataset.adj_matrix(data)

# Plot the ground truths
mapper = umap.UMAP().fit(data['features'])
umap.plot.points(mapper, labels = data['target'])

"""Calculates how accuracy of output compared to the target"""
def accuracy(output, target):
  preds = output.max(1)[1].type_as(target)
  correct = preds.eq(target).double()
  correct = correct.sum()
  return correct / len(target)

"""Trains the model once"""
def train(epoch):
  t = time.time()
  model.train()
  optimizer.zero_grad()
  # Run data through GCN layers and save the output
  output = model(features, adj)

  # Use negative log likelihood loss
  # Calculate loss and accuracy with training data
  loss_train = func.nll_loss(output[train_ds], target[train_ds])
  acc_train = accuracy(output[train_ds], target[train_ds])
  loss_train.backward()
  optimizer.step()

  # Calculate loss and accuracy with validation data
  loss_val = func.nll_loss(output[valid_ds], target[valid_ds])
  acc_val = accuracy(output[valid_ds], target[valid_ds])

  # Print out loss, accuracy values and epoch steps
  print('Epoch: {}/{}'.format(epoch+1, epochs),
        ' {:.4f}s'.format(time.time() - t),
        'loss: {:.4f}'.format(loss_train.item()),
        'accuracy: {:.4f}'.format(acc_train.item()),
        'val_loss: {:.4f}'.format(loss_val.item()),
        'val_acc: {:.4f}'.format(acc_val.item()))

"""Returns the GCN model"""
def model_init(features, data):
  model = modules.GCN(features.size(dim = 1),
                      features.size(dim = 1),
                      data['target'].argmax())
  return model

model = model_init(features, data)
optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=5e-4)

# Run training loop
epochs = 20
for epoch in range(epochs):
  train(epoch)
