
import torch
import torch.nn as nn
from tqdm import tqdm

def train(model,train_loader,val_loader,optimizer,scheduler,criterion,epochs, writer,device,test_loader):
    best_acc=0
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

        print(
        f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        model.eval()

        # These are used to record information in validation.
        valid_loss=[]
        valid_accs=[]

        # Iterate the validation set by batches.
        for batch in tqdm(val_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels=batch
            # print(imgs.shape)
            # imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits=model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss=criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc=(logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            # break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss=sum(valid_loss) / len(valid_loss)
        valid_acc=sum(valid_accs) / len(valid_accs)
        writer.add_scalar('Loss/val', valid_loss, epoch)
        writer.add_scalar('Accuracy/val',valid_acc, epoch)

        # Print the information.
        print(
            f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # save models
        if valid_acc > best_acc:
            print(
                f"Best model found at epoch {epoch}, saving model, pretrained_model.ckpt")
            # only save best to prevent output memory exceed error
            torch.save(model.state_dict(), "pretrained_model.ckpt")
            best_acc=valid_acc
    # test(model,test_loader,device)


def test(model, test_loader,device):
    """Predict the labels of the test data.
    Args:
        model: The model to be used for prediction.
        test_loader: The test data loader.
    Returns:
        The predicted labels.
    """
    model.eval()
    test_accs=[]
    # Iterate the testset by batches.
    for batch in tqdm(test_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels=batch
        # print(imgs.shape)
        # imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits=model(imgs.to(device))

        # We can still compute the loss (but not the gradient).

        # Compute the accuracy for current batch.
        acc=(logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        test_accs.append(acc)
        # break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_acc=sum(test_accs) / len(test_accs)

    # Print the information.
    print(
        f"[ Test ] acc = {valid_acc:.5f}")