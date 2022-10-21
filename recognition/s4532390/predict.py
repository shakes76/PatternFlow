from dataset import parse_data
from train import run_model

features, adjacency_matrix, targets = parse_data('recognition\\s4532390\\res\\facebook.npz')

run_model(features, adjacency_matrix, targets)
