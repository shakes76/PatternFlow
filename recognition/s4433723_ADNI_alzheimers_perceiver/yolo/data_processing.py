import cv2
import os

mask_img_dir = "ISIC_Train_Input/"
train_img_dir = "ISIC_Train_Masks/"

img = cv2.imread('lena.jpg')
mask = cv2.imread('mask.png',0)
res = cv2.bitwise_and(img,img,mask = mask)