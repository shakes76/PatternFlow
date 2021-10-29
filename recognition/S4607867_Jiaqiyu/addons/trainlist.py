# -*- coding: utf-8 -*-

import os
filePath = 'D:\\Data_science\\3710proj\\data\\train data\\'
trainlist= os.listdir(filePath)




output=[]
for path in trainlist:
    if 'txt' not in path:
        line='D:/Data_science/3710proj/data/train data/'+path
        output.append(line)
#print(output)

with open('ISIC_train.txt', 'w') as f:
    for i in range(len(output)):
        
        print(output[i])
        f.write(output[i]+"\n")
        
filePath = 'D:\\Data_science\\3710proj\\data\\validation data\\'
trainlist= os.listdir(filePath)




output=[]
for path in trainlist:
    if 'txt' not in path:
        line='D:/Data_science/3710proj/data/validation data/'+path
        output.append(line)
#print(output)

with open('ISIC_valid.txt', 'w') as f:
    for i in range(len(output)):
        
        print(output[i])
        f.write(output[i]+"\n")
        
          
filePath = 'D:\\Data_science\\3710proj\\data\\test data\\'
trainlist= os.listdir(filePath)      
output=[]
for path in trainlist:
    if 'txt' not in path:
        line='D:/Data_science/3710proj/data/test data/'+path
        output.append(line)
#print(output)

with open('ISIC_test.txt', 'w') as f:
    for i in range(len(output)):
        
        print(output[i])
        f.write(output[i]+"\n")