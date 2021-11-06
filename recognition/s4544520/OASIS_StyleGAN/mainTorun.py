import os
import torch
import argparse
from trainer2 import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face testing")
    # network init
    parser.add_argument("--batch_size", type=int, default=12, help="batch size")
    parser.add_argument("--nc", type=int, default=1, help="channel of output")
    parser.add_argument("--nz", type=int, default=512, help="dimension of the latent space")
    parser.add_argument("--init_size", type=int, default=4, help="the initial size of output")
    parser.add_argument("--size", type=int, default=128, help="the size of output")
    parser.add_argument("--stage_epoch", type=int, default=40, help="number of stage epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="initial lr")

    arg = parser.parse_args()
    trainer = Trainer(arg)
    trainer.train()
