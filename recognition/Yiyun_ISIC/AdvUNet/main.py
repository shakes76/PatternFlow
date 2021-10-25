"""Driver script for Improved UNet
"""
from model import AdvUNet

def main():
    model = AdvUNet()
    print(model.model.summary())
    

if __name__ == "__main__":
    main()
