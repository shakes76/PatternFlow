import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import train as tr
import torch
import sys

"""
show training effect curve.
"""


def show_acc_and_loss(train):
    plt.plot(train.val_acc_history, label='accuracy')
    plt.plot(train.loss_history, label='loss')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/loss')
    plt.ylim([0.2, 1.6])
    plt.legend(loc='lower right')
    plt.show()


"""
show TSNE visualization.
"""


def show_tsne(train):
    test_accuracy, test_data, test_labels = train.test(train.tensor_test_mask)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    embs = tsne.fit_transform(test_data)
    plt.title('tsne result')
    plt.scatter(embs[:, 0], embs[:, 1], marker='o', c=test_labels)
    plt.savefig("GCN_tsne.png")


def main():
    if len(sys.argv) == 1:
        train = tr.Train()
    else:
        train = tr.Train(sys.argv[1])
    train.train(show_result=False)
    show_tsne(train)  # show tsne.
    #show_acc_and_loss(train) # show accuracy and loss curve.


if __name__ == '__main__':
    print(torch.__version__)
    main()
