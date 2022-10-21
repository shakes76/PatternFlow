import sys
import torch
from model import Trainer

def main(arglist):
    """
    :param arglist:
        arglist[0]=dataroot, specifying the dataroot of training dataset
        arglist[1]=sample_path, specifying the directory to save the results including trained model, samples one .gif and one loss plot
    :return: None
    """
    dataroot=arglist[0]
    sample_path=arglist[1]
    #set the device
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    #set the trainer and paths
    t = Trainer(dataroot=dataroot, sample_path=sample_path,device=device)
    #the whole training loop
    t.training_loop()

if __name__ == "__main__":
    main(sys.argv[1:])