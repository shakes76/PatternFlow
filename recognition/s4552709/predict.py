def model_prediction(x_test):
    predict_y = model.predict(x_test, verbose=0)
    return predict_y

predict_y = model_prediction(x_test)
dice = dice_coefficient(y_seg_test, predict_y, smooth=0.0001) 
print("Dice coefficient is : ",dice)
