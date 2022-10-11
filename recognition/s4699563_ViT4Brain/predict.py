import torch
import torch.nn as nn
from tqdm import tqdm
def predict(model, test_loader,device):
    """Predict the labels of the test data.
    Args:
        model: The model to be used for prediction.
        test_loader: The test data loader.
    Returns:
        The predicted labels.
    """
    valid_accs=[]
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
        valid_accs.append(acc)
        # break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_acc=sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(
        f"[ Test ] acc = {valid_acc:.5f}")