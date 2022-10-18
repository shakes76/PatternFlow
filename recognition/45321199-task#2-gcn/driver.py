"""
    Driver/main script runner.
    Executes the full process.
"""

# from modules import Module
# from train import Trainer
# from predict import Predictor
from modules import GCN_Model
from dataset import DataLoader

def main():
    gcn = GCN_Model()
    gcn.create()
    gcn.compile()
    
    

if __name__ == "__main__":
    main()