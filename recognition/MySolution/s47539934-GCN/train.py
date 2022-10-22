import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn import preprocessing
def accuracy(output, labels):
    '''
    calculate accuracy
    parameters: 
    output:result of running an instance of the model
    labels: the true value
    function compares ratio of two values, giving result<1, 
    as predicted probability always less than true value
    '''
    predict = output.argmax(1)
    acc_ = torch.div(predict.eq(labels).sum(), labels.shape[0])
    return acc_
def loss(output,labels):

    prab = output.gather(1, labels.view(-1,1))
    loss = -torch.mean(prab)
    return loss
def train_model(n_epochs):
  '''
  parameter: number of epochs
  trains model over the range of the epoch and at each train, 
  calculates accuracy and losses
  '''
  train_losses=[]
  validation_losses=[]
  train_accuracies=[]
  validation_accuracies=[]
  for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    output=model(features,adj)
    train_loss=loss(output[train_set],labels[train_set])
    train_losses.append(train_loss.item())
   
    train_accuracy=accuracy(output[train_set],labels[train_set])
    train_accuracies.append(train_accuracy.item())
    train_loss.backward()
    optimizer.step()
    output=model(features,adj)
    validation_loss=loss(output[val_set],labels[val_set])
    validation_losses.append(validation_loss.item())
    validation_accuracy=accuracy(output[val_set],labels[val_set])
    validation_accuracies.append(validation_accuracy.item())
    print('Epoch: {:04d}'.format(epoch + 1),
              'Train loss: {:.4f}'.format(train_loss.item()),
              'Train accuracy: {:.4f}'.format(train_accuracy.item()),
              'Validation loss: {:.4f}'.format(validation_loss.item()),
              'Validation accuracy: {:.4f}'.format(validation_accuracy.item()))
    torch.save(model.state_dict(),'train_model.pth')#this random file just used as buffer
  return train_accuracies,validation_accuracies
#test
def test_mode():
  model.load_state_dict(torch.load('train_mode.pth'))
  output=model(features,adj)
  test_loss=loss(output[test_set],labels[test_set])
  test_accuracy=accuracy(output[test_set],labels[test_set])
  print('Test set results:',
        'Test loss: {:.4f}'.format(test_loss.item()),
        'Test accuracy: {:.4f}'.format(test_accuracy.item()))
