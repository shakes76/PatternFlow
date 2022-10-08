
import torch
import torch.nn as nn
from tqdm import tqdm

def train(model,train_loader,val_loader,optimizer,scheduler,criterion,epochs, writer,device):
    for epoch in range(epochs):

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss=[]
        train_accs=[]

        for batch in tqdm(train_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels=batch
            # imgs = imgs.half()
            # print(imgs.shape,labels.shape)

            # Forward the data. (Make sure data and model are on the same device.)
            # print(imgs.shape,356) #batch,channel,img_size,img_size
            # imgs=imgs.permute(0,2,3,1)
            # print(imgs.shape,356) #batch,channel,img_size,img_size

            logits=model(imgs.to(device))

            # # Calculate the cross-entropy loss.
            # # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss=criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm=nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc=(logits.argmax(dim=-1) == labels.to(device)).float().mean()
            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
            
        scheduler.step()

        train_loss=sum(train_loss) / len(train_loss)
        train_acc=sum(train_accs) / len(train_accs)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train',train_acc, epoch)