'''
Author_name: Arsh Upadhyaya
roll no. s4753993
to plot classes of dataset facebook.npz and see how same classes come together in a cluster
'''
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
def plot_tsne(labels,output):
  tsne=TSNE().fit_transform(outputs)
  plt.title('tsne result')
  plt.scatter(tsne[:,0],tsne[:,1],marker='o',c=labels)
  plt.savefig("GCS_tsne.png")
