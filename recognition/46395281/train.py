import sys
import torch
from model import Trainer
def main(arglist):
    dataroot=arglist[0]
    sample_path=arglist[1]
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    t = Trainer(dataroot=dataroot, sample_path=sample_path,device=device)
    t.training_loop()
if __name__ == "__main__":
    main(sys.argv[1:])