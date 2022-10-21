from dataset import parse_data
from train import run_model
import matplotlib.pyplot as plt

# Author - Robert Francis 45323906

if __name__ == "__main__":
    # Parse facebook.npz to get features, adjacency matrix and targets
    features, adjacency_matrix, targets = parse_data('recognition\\s4532390\\res\\facebook.npz')

    # Run the model - builds and tests - set tsne plot to true
    run_model(features, adjacency_matrix, targets, tsne_plot=True)

    # Show any graphs
    plt.show()
    