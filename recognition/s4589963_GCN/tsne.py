import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import train as tr


def main():
    train = tr.Train()
    train.train(show_result=False)
    test_accuracy, test_data, test_labels = train.test(train.tensor_test_mask)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    embs = tsne.fit_transform(test_data)
    plt.title('tsne result')
    plt.scatter(embs[:, 0], embs[:, 1], marker='o', c=test_labels)
    plt.savefig("GCN_tsne.png")


if __name__ == '__main__':
    main()
