"""
 iterates through directories, and does train test split. 
 Saves labels to .txt files so same splits on goliath / local machine can 
 be loaded from the .txt lists.

"""

import os
import numpy as np
import json

START_DIR = "data/AKOA_Analysis/"
START_DIR = "data/resize/"


VALIDATION_SPLIT = 0.20
TEST_SPLIT = 0.40

train_labels = ""
validation_labels = ""
test_labels = ""
label_to_content = {}

label = 0
for dir in os.listdir(START_DIR):
    for file in os.listdir(START_DIR + "/" + dir):
        filename = f"{dir}/{file}"
        r = np.random.rand()
        if r > VALIDATION_SPLIT:
            train_labels += f"{filename} {label}\n"
        elif r > TEST_SPLIT:
            test_labels += f"{filename} {label}\n"
        else:
            validation_labels += f"{filename} {label}\n"
    label_to_content[str(label)] = dir
    label += 1

with open("train.txt", "w") as f:
    f.write(train_labels)    

with open("test.txt", "w") as f:
    f.write(test_labels)   

with open("validation.txt", "w") as f:
    f.write(validation_labels)    

with open("label_to_content.json", 'w') as f:
    json.dump(label_to_content, f, indent=4)