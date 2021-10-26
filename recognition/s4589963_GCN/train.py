import dataset_loader as loader
import numpy as np
import torch
from sklearn.preprocessing import LabelBinarizer
from model import GCN
import time
import sys

PATH = "F:\\3710report\\facebook.npz"

"""
Main training process.
"""


class Train:
    """
    initialise the train process.
    """

    def __init__(self, path=PATH):
        self.val_acc_history = []
        self.loss_history = []
        adjacency, features, labels = loader.load_data(path)
        # encoded the label to one hot
        one_hot = LabelBinarizer()
        labels = one_hot.fit_transform(labels)

        # normalize adjacency matrix and features
        adjacency = loader.normalize_adj(adjacency)
        features = loader.normalize_features(features)

        # transform to tensor.
        features = torch.FloatTensor(np.array(features))
        labels = torch.LongTensor(np.where(labels)[1])

        node_number = features.shape[0]

        # get train mask, validation mask and test mask. (split the data set)
        self.train_mask = np.zeros(node_number, dtype=bool)
        self.val_mask = np.zeros(node_number, dtype=bool)
        self.test_mask = np.zeros(node_number, dtype=bool)
        self.init_masks()

        self.learning_rate = 0.1  # learning rate.
        self.weight_decay = 5e-4  # weight decay to prevent overfitting.
        self.epochs = 200  # we train 200 epochs.

        self.device = "cpu"  # we don't have gpu so we use cpu as our device.

        self.model = GCN().to(self.device)  # Initialise the model and load it to device.

        # initialise loss function and optimizer.
        # we use CrossEntropyLoss as our loss function and adam as our optimizer/
        self.loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                          weight_decay=self.weight_decay)

        # load features and labels to device.
        self.tensor_features = features.to(self.device)
        self.tensor_labels = labels.to(self.device)

        # Transform masks to pytorch tensor and load them to cpu.
        self.tensor_train_mask = torch.from_numpy(self.train_mask).to(self.device)
        self.tensor_val_mask = torch.from_numpy(self.val_mask).to(self.device)
        self.tensor_test_mask = torch.from_numpy(self.test_mask).to(self.device)

        # initialise the adjacency matrix tensor and load it to cpu.
        indices = torch.from_numpy(np.asarray([adjacency.row, adjacency.col]).astype('int64')).long()
        values = torch.from_numpy(adjacency.data.astype(np.float32))
        # we have 22470 data in this dataset.
        self.tensor_adjacency = torch.sparse.FloatTensor(indices, values, (22470, 22470)).to(self.device)

    """
    initialise masks.
    """

    def init_masks(self):
        self.train_mask[:3000] = True
        self.val_mask[3000:4000] = True
        self.test_mask[4000:5000] = True

    """
    test the model to know the accuarcy on corresponding mask.
    """

    def test(self, mask):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.tensor_adjacency, self.tensor_features)
            test_mask_logits = logits[mask]
            predict_labels = test_mask_logits.max(1)[1]
            accuracy = torch.eq(predict_labels, self.tensor_labels[mask]).float().mean()
        return accuracy, test_mask_logits.cpu().numpy(), self.tensor_labels[mask].cpu().numpy()

    """
    main training process.
    """

    def train(self, show_result=True):
        if show_result:
            print("======================================================================================")
        self.model.train()
        train_labels = self.tensor_labels[self.tensor_train_mask]

        for epoch in range(1, self.epochs + 1):
            start = time.time()

            logits = self.model(self.tensor_adjacency, self.tensor_features)  # Forward propagation.
            train_mask_logits = logits[
                self.tensor_train_mask]  # Only nodes on training set are selected for supervision.
            loss_val = self.loss(train_mask_logits, train_labels)  # Calculate loss value.
            self.optimizer.zero_grad()
            loss_val.backward()  # Back propagation to calculate the gradient of parameters.
            self.optimizer.step()  # Use optimizer to update gradient.

            train_acc, _, _ = self.test(self.tensor_train_mask)  # Calculate the accuracy on the training set.
            val_acc, _, _ = self.test(self.tensor_val_mask)  # Calculate the accuracy on the validation set.

            """
            Record the changes of loss value and accuracy in 
            the training process for visualization.
            """
            self.loss_history.append(loss_val.item())
            self.val_acc_history.append(val_acc.item())
            if show_result:
                # print training result every 10 epochs.
                print("Epoch %3d/%d: Loss %.4f, Train_accuracy %.4f, Validation_accuracy %.4f, Time %.4f" % (
                    epoch, self.epochs, loss_val.item(), train_acc.item(), val_acc.item(), time.time() - start))


def main():
    if len(sys.argv) == 1:
        train = Train()
    else:
        train = Train(sys.argv[1])
    train = Train()
    train.train(show_result=True)


if __name__ == '__main__':
    main()
