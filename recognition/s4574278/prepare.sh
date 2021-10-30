#!/bin/bash

# VAR
STORE=autodl-nas
WORKSPACE=autodl-tmp

# PREPARE workspace
unzip ./$STORE/ISIC2018_Task1-2_Training_Data_uq.zip -d ./$STORE
cp ./$STORE/ISIC2018_Task1_Training_GroundTruth_x2 ./$WORKSPACE/ground_truth
cp ./$STORE/ISIC2018_Task1-2_Training_Input_x2 ./$WORKSPACEp/input
mkdir ./$WORKSPACE/annotation

# Extract annotation and do preprocessing
jupyter nbconvert --execute --to data-result.ipynb data.ipynb