#!/usr/bin/env python3

"""
    Driver script and entry point for Perceiver transformer
    Invokes the data processing script, data.py and 
    the model itself -- model.py.

    References: http://arxiv.org/abs/2103.03206
"""

__author__ = "Chegne Eu Joe"
__email__ = "e.chegne@uqconnect.edu.au"

# Argparse - Argument parsing
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Perceiver transformer on ADNI brain data"
    )

    # Create flags arguments for driver script flexibility
    parser.add_argument(
        "dataset_path", type=str, help="Path to dataset (ADNI brain data)"
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=30,
        help="The number of epochs the algorithm will train on",
    )

    args = parser.parse_args()
    print(f"Num Epochs: args.epochs")

    ## INVOKE MODEL AND DATA PROCESSING


if __name__ == "__main__":
    main()
