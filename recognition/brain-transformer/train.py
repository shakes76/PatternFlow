import matplotlib.pyplot as plt
import vit_pytorch
import torch
import numpy as np
import dataset
import modules


train_data, valid_data = dataset.torch_train('train')
test_data = dataset.torch_test('test')

model, optimizer, criterion, scheduler = modules.build_model(
    dim=1024, 
    image_size=256, 
    patch_size=32,
    num_classes=2,
    depth=12,
    heads=16,
    mlp_dim=2048,
    channels=1,
    dropout=0.4,
    emb_dropout=0.4,
    lr = 0.0001
    )
model = model.cuda()

EPOCHS = 30

train_iter = iter(train_data)
train_size = 0
valid_iter = iter(valid_data)
valid_size = 0
for i in train_iter:
    train_size += i[0].shape[0]
for i in valid_iter:
    valid_size += i[0].shape[0]
min_valid_loss = np.inf

history = {'train_loss':[0], 'train_acc':[0],'valid_loss':[0],'valid_acc':[0]}
for epoch in range(EPOCHS):
    train_loss = 0
    train_acc = 0
    train_iter = iter(train_data)
    model.train()
    for batch, labels in train_iter:
        batch, labels = batch.cuda(), labels.cuda()
        optimizer.zero_grad()
        prediction = model(batch)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()
        acc = sum(torch.argmax(prediction,dim=1) == torch.argmax(labels,dim=1)).cpu().detach().numpy()
        train_acc += acc
        train_loss += loss.cpu().detach().numpy() * len(batch)/train_size
    train_acc /= train_size

    valid_loss = 0
    valid_acc = 0
    model.eval()
    valid_iter = iter(valid_data)
    test_valid_size = 0
    for batch, labels in valid_iter:
        test_valid_size+= labels.shape[0]
        batch, labels = batch.cuda(),labels.cuda()
        prediction = model(batch)
        loss = criterion(prediction, labels)
        acc = sum(torch.argmax(prediction,dim=1) == torch.argmax(labels,dim=1)).item()
        valid_acc += acc
        valid_loss += loss.cpu().detach().numpy()*len(batch)/valid_size
    valid_acc /= valid_size
    # scheduler.step()
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['valid_acc'].append(valid_acc)
    history['valid_loss'].append(valid_loss)

    print(f"Epoch: {epoch+1}\nTrain loss: {train_loss}\nTrain Accuracy: {train_acc}\nValid Loss: {valid_loss}\nValid Accuracy: {valid_acc}\n")
    
    if min_valid_loss > valid_loss:
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved/best_model.pth')


plt.figure(figsize=(12, 8), dpi=80)
plt.plot(history['train_acc'])
plt.plot(history['valid_acc'])
plt.xlim([0,20])
plt.xticks([0,5,10,15,20])
plt.ylim([0,1])
plt.title('Accuracy')
plt.legend(['Training', 'Validation'])
plt.show()

plt.figure(figsize=(12, 8), dpi=80)
plt.plot(history['train_loss'])
plt.plot(history['valid_loss'])
plt.xlim([0,20])
plt.xticks([0,5,10,15,20])
plt.ylim([0,1])
plt.title('Loss')
plt.legend(['Training', 'Validation'])
plt.show()
