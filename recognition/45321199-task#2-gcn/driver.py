"""
    Driver/main script runner.
    Executes the full process.
"""

# from modules import Module
# from train import Trainer
# from predict import Predictor
from train import Trainer

def main():
    trainer = Trainer()
    trainer.train()
    
    

if __name__ == "__main__":
    main()