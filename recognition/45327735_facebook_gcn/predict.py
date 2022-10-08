"""
Example usage of the GCN model applied to the Facebook dataset.

Created on Fri Oct 07 12:48:34 2022

@author: Crispian Yeomans
"""
from dataset import Dataset

def main():
    path = "C:\\Users\\cyeol\\Documents\\University\\2022\\COMP3710"
    filename = "facebook"
    dataset = Dataset(path, filename)
    dataset.summary(3)
    print(dataset.get_tensors())

if __name__ == "__main__":
    main()