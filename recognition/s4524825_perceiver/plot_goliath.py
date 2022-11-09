"""
Plots training history from csv log written by goliath process
"""

import os 
import numpy as np 
import matplotlib.pyplot as plt 
import csv 

TRAIN_HISTORY = "goliath/train_history_.csv"
TEST_HISTORY = "goliath/test_history__.csv"


def get_loss_acc_from_csv(filename):
    acc, loss = [], []
    with open(filename, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            if len(row) >= 2:
                acc.append(float(row[1]))
                loss.append(float(row[2]))

    return acc, loss

train_acc, train_loss = get_loss_acc_from_csv(TRAIN_HISTORY)
test_acc, test_loss = get_loss_acc_from_csv(TEST_HISTORY)

plt.plot(train_acc)
plt.plot(test_acc)
plt.legend(["Train", "Test"])
plt.title("Accuracy")

plt.figure()
plt.plot(train_loss)
plt.plot(test_loss)
plt.legend(["Train", "Test"])
plt.title("Loss")

plt.show()