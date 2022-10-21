from train import train
from predict import predict

#dset_path = "/root/.keras/datasets/ISIC-2017_Training_Data"
#mask_path = "/root/.keras/datasets/ISIC-2017_Training_Part1_GroundTruth"

model, results, X_val, Y_val = train(dset_path, mask_path)
predict(model, results, X_val, Y_val)