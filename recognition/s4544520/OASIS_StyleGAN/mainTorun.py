import os
import torch
import argparse
from trainer2 import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face testing")
    # network init
    parser.add_argument("--batch_size", type=int, default=12, help="batch size")
    parser.add_argument("--nc", type=int, default=1, help="number of channels of the generated image")
    parser.add_argument("--nz", type=int, default=512, help="dimension of the input noise")
    parser.add_argument("--init_size", type=int, default=4, help="the initial size of the generated image")
    parser.add_argument("--size", type=int, default=128, help="the final size of the generated image")
    parser.add_argument("--stage_epoch", type=int, default=40, help="number of transition epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")

    arg = parser.parse_args()
    trainer = Trainer(arg)
    trainer.train()
