"""
    Driver/main script runner.
    Executes the full process.
"""

# from modules import Module
# from train import Trainer
# from predict import Predictor
from dataset import DataLoader

def main():
    data_loader = DataLoader()
    data_loader.parse_data()
    

if __name__ == "__main__":
    main()