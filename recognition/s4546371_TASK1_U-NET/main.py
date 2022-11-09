from data_cleaning import*
from model import *
from dice_cofficient import *


if __name__ == '__main__':

    ###import data from files
    path_ground_truth="ISIC2018_Task1_Training_GroundTruth_x2/"
    path_train="ISIC2018_Task1-2_Training_Input_x2/"
    truth = load_data(path_ground_truth)[1:]
    train = load_data(path_train)[1:]
    X = read_data(train[0:-1],2) ##exclude some txt in floder
    y = read_data(truth[0:-1],1) ##exclude some txt in floder
    
    ### shuffle data and split the data to training set,testing set,validation set
    X_train, X_val,X_test,y_train,y_val,y_test = shuffle_data(X,y)
    y_train=creat_mask(y_train)
    y_val=creat_mask(y_val)
    y_test=creat_mask(y_test)
    
    
    ###normalize data
    X_train = np.array(X_train)/255
    X_val = np.array(X_val)/255
    X_test = np.array(X_test)/255
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    
    
    #print(np.array(X_train).shape)
    #print(np.array(X_val).shape)
    #print(np.array(X_test).shape)
    #print(np.array(y_train).shape)
    #print(np.array(y_val).shape)
    #print(np.array(y_test).shape)
    
    ###training the model
    model=improved_unet_a(input_size=(256, 256, 3),output_channels=2)
    model.compile(optimizer = Adam(),loss='binary_crossentropy', metrics = dice_coefficient)
    history = model.fit(X_train,y_train, epochs=8, batch_size=8, validation_data = (X_val,y_val))
    
    
    ###save the histroy during the training model
    dice_coefficient=history.history["dice_coefficient"]
    loss=history.history["loss"]
    val_loss=history.history["val_loss"]
    val_dice_coefficient=history.history["val_dice_coefficient"]
    np_dice=np.array(dice_coefficient).reshape((1,len(dice_coefficient)))
    np_loss=np.array(loss).reshape((1,len(loss)))
    np_val_dice=np.array(val_dice_coefficient).reshape((1,len(val_dice_coefficient)))
    np_val_loss=np.array(val_loss).reshape((1,len(val_loss)))
    
    np_out_training =np.concatenate([np_dice,np_loss],axis=0)
    np_out_val=np.concatenate([np_val_dice,np_val_loss],axis=0)
    np.savetxt("save_training.txt",np_out_training)
    np.savetxt("svae_val.txt",np_out_val)

    print(model.evaluate(X_test,y_test))
    
    
    
    
    #!mkdir -p saved_model
    model.save('unet_final')
