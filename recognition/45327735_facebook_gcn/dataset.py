"""
The data loader for loading and pre-processing the Facebook Large Page-Page Network.

The loaded dataset should be the partially pre-processed dataset in .NPZ format ("facebook.NPZ").

Labels:
0 = tvshow
1 = company
2 = government
3 = politician

Created on Fri Oct 07 12:48:34 2022

@author: Crispian Yeomans
"""

from numpy import load

class Dataset:
    """Represents and preprocesses the Facebook dataset"""
    def __init__(self, path, filename='facebook'):
        self.data_numpy = self._load(path, filename)

    def _load(self, path, filename):
        """Loads the partially preprocessed .npz Facebook dataset"""
        return load(path+"\\"+filename+'.npz')

    def summary(self, n=5):
        """
        Prints a summary of the .npz dataset

        :params n: number of data points to print
        """
        data = self.data_numpy

        # Summary data
        print("\nSUMMARY OF EDGES")
        for i in range(0, n):
            print("Edges:", data["edges"][i])

        print("\nSUMMARY OF NODES")
        for i in range(0, n):
            print("\nNode", i, ":\n")
            print("\nFeatures:", data["features"][i])
            print("Target:", data["target"][i])

        # Summary of dataset
        print("\nSummary",
              "\nNum of Edges:", len(data["edges"]), "/ 2",
              "\nNum of Nodes", len(data["features"][0]),
              "\nNum of Targets:", 4)
        for i in range(0, n):
            print("Node:", i,
                  "Num of Features:", len(data["features"][i]),
                  "Target:", data["target"][i])