from Data_prepare import *
from GCN_Layer import *
from  GCN_model_define import *
from GCN_test_train import *
# this is a script to used in final running
# training model process / will print accuracy about loss and Validation and training
train()


# result visualization as a plot use TSNE method.

from sklearn.manifold import TSNE
# the result is the classfy results, which can be present as 4 classes
test_accuracy, test_data, test_labels = test(tensor_test_mask)
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
low_dim_embs = tsne.fit_transform(test_data)
plt.title('TSNE result')
plt.scatter(low_dim_embs[:,0], low_dim_embs[:,1], marker='o', c=test_labels)
plt.savefig("tsne.png")
plt.show()
