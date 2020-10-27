#!/usr/bin/env/python

"""
Driver script for ISICs UNet recognition problem.

Created by Christopher Bailey (45576430) for COMP3710 Report.
"""


import tensorflow as tf
from isicsunet import IsicsUnet

def main():
    print(tf.__version__)

    model = IsicsUnet()
    model.load_data()

    print("END")


if __name__ == "__main__":
    main()
