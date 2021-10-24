import dataset_loader as loader
import numpy as np
import torch
from sklearn.preprocessing import LabelBinarizer
from model import GCN
import time

PATH = "F:\\3710report\\facebook.npz"


adjacency, features, labels = loader.load_data(PATH)

# encoded the label to one hot
onehot = LabelBinarizer()
labels = onehot.fit_transform(labels)

# normalize adjacency matrix and features
adjacency = loader.normalize_adj(adjacency)
features = loader.normalize_features(features)

# transform to tensor.
features = torch.FloatTensor(np.array(features))
labels = torch.LongTensor(np.where(labels)[1])

print(adjacency)
print(features)
print(labels)

node_number = features.shape[0]

# get train mask, validation mask and test mask. (split the data set)
train_mask = np.zeros(node_number, dtype=bool)
val_mask = np.zeros(node_number, dtype=bool)
test_mask = np.zeros(node_number, dtype=bool)
train_mask[:3000] = True
val_mask[3000:4000] = True
test_mask[4000:5000] = True

learning_rate = 0.1  # learning rate.
weight_decay = 5e-4  # weight decay to prevent overfitting.
epochs = 200  # we train 200 epochs.

device = "cpu"  # we don't have gpu so we use cpu as our device.
model = GCN().to(device)  # Initialise the model and load it to device.

# initialise loss function and optimizer.
# we use CrossEntropyLoss as our loss function and adam as our optimizer/
loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# load features and labels to device.
tensor_features = features.to(device)
tensor_labels = labels.to(device)

# Transform masks to pytorch tensor and load them to cpu.
tensor_train_mask = torch.from_numpy(train_mask).to(device)
tensor_val_mask = torch.from_numpy(val_mask).to(device)
tensor_test_mask = torch.from_numpy(test_mask).to(device)

# initialise the adjacency matrix tensor and load it to cpu.
indices = torch.from_numpy(np.asarray([adjacency.row, adjacency.col]).astype('int64')).long()
values = torch.from_numpy(adjacency.data.astype(np.float32))
# we have 22470 data in this dataset.
tensor_adjacency = torch.sparse.FloatTensor(indices, values, (22470, 22470)).to(device)

"""
test the model to know the accuarcy on corresponding mask.
"""


def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_features)
        test_mask_logits = logits[mask]
        predict_labels = test_mask_logits.max(1)[1]
        accuarcy = torch.eq(predict_labels, tensor_labels[mask]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_labels[mask].cpu().numpy()


"""
main training process.
"""


def train():
    print("======================================================================================")
    loss_history = []
    val_acc_history = []
    model.train()
    train_labels = tensor_labels[tensor_train_mask]

    for epoch in range(1, epochs + 1):
        start = time.time()

        logits = model(tensor_adjacency, tensor_features)  # Forward propagation.
        train_mask_logits = logits[tensor_train_mask]  # Only nodes on training set are selected for supervision.
        loss_val = loss(train_mask_logits, train_labels)  # Calculate loss value.
        optimizer.zero_grad()
        loss_val.backward()  # Back propagation to calculate the gradient of parameters.
        optimizer.step()  # Use optimizer to update gradient.

        train_acc, _, _ = test(tensor_train_mask)  # Calculate the accuracy on the training set.
        val_acc, _, _ = test(tensor_val_mask)  # Calculate the accuracy on the validation set.

        """
        Record the changes of loss value and accuracy in 
        the training process for visualization.
        """
        loss_history.append(loss_val.item())
        val_acc_history.append(val_acc.item())
        # print training result every 10 epochs.
        print("Epoch %3d/%d: Loss %.4f, Train_accuracy %.4f, Validation_accuracy %.4f, Time %.4f" % (
            epoch, epochs, loss_val.item(), train_acc.item(), val_acc.item(), time.time() - start))


def main():
    train()


if __name__ == '__main__':
    main()
