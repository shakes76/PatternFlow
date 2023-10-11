#Import the GCN model.
from GCN_train import *
from sklearn.manifold import TSNE
from PIL import Image

def tsne():
    test_accuracy, test_data, test_labels = train.test(tensor_test_mask)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=10000)
    low_dim_embs = tsne.fit_transform(test_data)
    plt.title('GCN tsne result(3710)')
    plt.scatter(low_dim_embs[:,0], low_dim_embs[:,1], marker='o', c=test_labels)
    #Save the image.
    plt.savefig("tsne.png")

def result():
    im = Image.open("tsne.png")
    im.show()

# Train model class and draw the accuracy and loss.
train = Train()
train.train()
train.acc()
# TSNE embeddings plot
tsne()
# The TSNE result image.
result()
