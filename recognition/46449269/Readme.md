# Segment the ISICs data set with the Improved UNet with all labels having a minimum Dice similarity coefficient of 0.8 on the test set.

COMP3710 Report Task 1


Ge Song 4644926

## Approach

The UNet model consists of downsampling part for feature learning and upsampling part for mask segmentation,the result of each layer in the downsampling part is also concatenated to the corresponding layer in the upsampling part before convolution, achieving better segmentation performance.

## Result and plot

The input images, which originally has varying sizes were resized to (256 x 256) for faster training. The training data was split between training:validation:testing on 60:20:20 ratio, which seems to be a commonly prescribed starting point for smaller datasets. Batch size of 8 and Adam optimizer learning rate of 0.00001 was used, which gives satisfactory results, a validation DSC of 0.7433 with epoch =7, but drop to 0.61, when set epoch = 50, given dsc = 0.7001. 


```python
Epoch 1/50
156/156 [==============================] - 209s 1s/step - loss: 0.3339 - accuracy: 0.8328 - dice_coef: 0.6664 - val_loss: 0.3172 - val_accuracy: 0.8413 - val_dice_coef: 0.6829
Epoch 2/50
156/156 [==============================] - 207s 1s/step - loss: 0.3257 - accuracy: 0.8385 - dice_coef: 0.6740 - val_loss: 0.2987 - val_accuracy: 0.8597 - val_dice_coef: 0.7011
Epoch 3/50
156/156 [==============================] - 207s 1s/step - loss: 0.3129 - accuracy: 0.8558 - dice_coef: 0.6868 - val_loss: 0.2849 - val_accuracy: 0.8716 - val_dice_coef: 0.7151
Epoch 4/50
156/156 [==============================] - 207s 1s/step - loss: 0.3054 - accuracy: 0.8545 - dice_coef: 0.6945 - val_loss: 0.2968 - val_accuracy: 0.8423 - val_dice_coef: 0.7033
Epoch 5/50
156/156 [==============================] - 207s 1s/step - loss: 0.2918 - accuracy: 0.8564 - dice_coef: 0.7082 - val_loss: 0.2911 - val_accuracy: 0.8402 - val_dice_coef: 0.7091
Epoch 6/50
156/156 [==============================] - 208s 1s/step - loss: 0.2739 - accuracy: 0.8603 - dice_coef: 0.7264 - val_loss: 0.2266 - val_accuracy: 0.8890 - val_dice_coef: 0.7736
Epoch 7/50
156/156 [==============================] - 207s 1s/step - loss: 0.2571 - accuracy: 0.8697 - dice_coef: 0.7433 - val_loss: 0.2015 - val_accuracy: 0.9045 - val_dice_coef: 0.7984
Epoch 8/50
156/156 [==============================] - 207s 1s/step - loss: 0.3057 - accuracy: 0.8545 - dice_coef: 0.6939 - val_loss: 0.3809 - val_accuracy: 0.7849 - val_dice_coef: 0.6189
Epoch 9/50
156/156 [==============================] - 206s 1s/step - loss: 0.3863 - accuracy: 0.8033 - dice_coef: 0.6141 - val_loss: 0.3775 - val_accuracy: 0.8336 - val_dice_coef: 0.6224
Epoch 10/50
156/156 [==============================] - 206s 1s/step - loss: 0.3798 - accuracy: 0.8114 - dice_coef: 0.6196 - val_loss: 0.3747 - val_accuracy: 0.8389 - val_dice_coef: 0.6251
Epoch 11/50
156/156 [==============================] - 206s 1s/step - loss: 0.3823 - accuracy: 0.8106 - dice_coef: 0.6180 - val_loss: 0.3622 - val_accuracy: 0.8461 - val_dice_coef: 0.6380
Epoch 12/50
156/156 [==============================] - 205s 1s/step - loss: 0.3810 - accuracy: 0.8132 - dice_coef: 0.6193 - val_loss: 0.3653 - val_accuracy: 0.8136 - val_dice_coef: 0.6345
Epoch 13/50
156/156 [==============================] - 205s 1s/step - loss: 0.3811 - accuracy: 0.8186 - dice_coef: 0.6185 - val_loss: 0.3778 - val_accuracy: 0.7852 - val_dice_coef: 0.6222
Epoch 14/50
156/156 [==============================] - 205s 1s/step - loss: 0.3803 - accuracy: 0.8105 - dice_coef: 0.6189 - val_loss: 0.3774 - val_accuracy: 0.8377 - val_dice_coef: 0.6225
Epoch 15/50
156/156 [==============================] - 205s 1s/step - loss: 0.3810 - accuracy: 0.8176 - dice_coef: 0.6189 - val_loss: 0.3624 - val_accuracy: 0.8329 - val_dice_coef: 0.6377
Epoch 16/50
156/156 [==============================] - 205s 1s/step - loss: 0.3723 - accuracy: 0.8221 - dice_coef: 0.6282 - val_loss: 0.3821 - val_accuracy: 0.7823 - val_dice_coef: 0.6180
Epoch 17/50
156/156 [==============================] - 205s 1s/step - loss: 0.3736 - accuracy: 0.8244 - dice_coef: 0.6265 - val_loss: 0.3694 - val_accuracy: 0.8463 - val_dice_coef: 0.6306
Epoch 18/50
156/156 [==============================] - 205s 1s/step - loss: 0.3786 - accuracy: 0.8228 - dice_coef: 0.6213 - val_loss: 0.3642 - val_accuracy: 0.8461 - val_dice_coef: 0.6359
Epoch 19/50
156/156 [==============================] - 205s 1s/step - loss: 0.3718 - accuracy: 0.8263 - dice_coef: 0.6277 - val_loss: 0.3554 - val_accuracy: 0.8471 - val_dice_coef: 0.6444
Epoch 20/50
156/156 [==============================] - 205s 1s/step - loss: 0.3660 - accuracy: 0.8294 - dice_coef: 0.6339 - val_loss: 0.3419 - val_accuracy: 0.8508 - val_dice_coef: 0.6581
Epoch 21/50
156/156 [==============================] - 205s 1s/step - loss: 0.3599 - accuracy: 0.8283 - dice_coef: 0.6402 - val_loss: 0.3697 - val_accuracy: 0.8496 - val_dice_coef: 0.6302
Epoch 22/50
156/156 [==============================] - 205s 1s/step - loss: 0.3651 - accuracy: 0.8278 - dice_coef: 0.6353 - val_loss: 0.3478 - val_accuracy: 0.8302 - val_dice_coef: 0.6518
Epoch 23/50
156/156 [==============================] - 206s 1s/step - loss: 0.3572 - accuracy: 0.8346 - dice_coef: 0.6431 - val_loss: 0.3393 - val_accuracy: 0.8529 - val_dice_coef: 0.6608
Epoch 24/50
156/156 [==============================] - 205s 1s/step - loss: 0.3517 - accuracy: 0.8299 - dice_coef: 0.6482 - val_loss: 0.3415 - val_accuracy: 0.8309 - val_dice_coef: 0.6586
Epoch 25/50
156/156 [==============================] - 206s 1s/step - loss: 0.3529 - accuracy: 0.8314 - dice_coef: 0.6467 - val_loss: 0.3278 - val_accuracy: 0.8481 - val_dice_coef: 0.6722
Epoch 26/50
156/156 [==============================] - 206s 1s/step - loss: 0.3512 - accuracy: 0.8321 - dice_coef: 0.6491 - val_loss: 0.3470 - val_accuracy: 0.8408 - val_dice_coef: 0.6530
Epoch 27/50
156/156 [==============================] - 205s 1s/step - loss: 0.3575 - accuracy: 0.8316 - dice_coef: 0.6426 - val_loss: 0.3375 - val_accuracy: 0.8303 - val_dice_coef: 0.6624
Epoch 28/50
156/156 [==============================] - 206s 1s/step - loss: 0.3534 - accuracy: 0.8296 - dice_coef: 0.6462 - val_loss: 0.3361 - val_accuracy: 0.8319 - val_dice_coef: 0.6641
Epoch 29/50
156/156 [==============================] - 205s 1s/step - loss: 0.3442 - accuracy: 0.8360 - dice_coef: 0.6562 - val_loss: 0.3126 - val_accuracy: 0.8540 - val_dice_coef: 0.6872
Epoch 30/50
156/156 [==============================] - 206s 1s/step - loss: 0.3393 - accuracy: 0.8356 - dice_coef: 0.6604 - val_loss: 0.3208 - val_accuracy: 0.8308 - val_dice_coef: 0.6794
Epoch 31/50
156/156 [==============================] - 206s 1s/step - loss: 0.3372 - accuracy: 0.8375 - dice_coef: 0.6631 - val_loss: 0.3267 - val_accuracy: 0.8189 - val_dice_coef: 0.6733
Epoch 32/50
156/156 [==============================] - 206s 1s/step - loss: 0.3336 - accuracy: 0.8348 - dice_coef: 0.6665 - val_loss: 0.3188 - val_accuracy: 0.8580 - val_dice_coef: 0.6808
Epoch 33/50
156/156 [==============================] - 206s 1s/step - loss: 0.3275 - accuracy: 0.8408 - dice_coef: 0.6725 - val_loss: 0.3012 - val_accuracy: 0.8529 - val_dice_coef: 0.6986
Epoch 34/50
156/156 [==============================] - 206s 1s/step - loss: 0.3216 - accuracy: 0.8404 - dice_coef: 0.6784 - val_loss: 0.3298 - val_accuracy: 0.8654 - val_dice_coef: 0.6701
Epoch 35/50
156/156 [==============================] - 205s 1s/step - loss: 0.3267 - accuracy: 0.8416 - dice_coef: 0.6730 - val_loss: 0.3167 - val_accuracy: 0.8392 - val_dice_coef: 0.6832
Epoch 36/50
156/156 [==============================] - 206s 1s/step - loss: 0.3184 - accuracy: 0.8418 - dice_coef: 0.6815 - val_loss: 0.2981 - val_accuracy: 0.8585 - val_dice_coef: 0.7019
Epoch 37/50
156/156 [==============================] - 206s 1s/step - loss: 0.3171 - accuracy: 0.8446 - dice_coef: 0.6830 - val_loss: 0.3027 - val_accuracy: 0.8533 - val_dice_coef: 0.6973
Epoch 38/50
156/156 [==============================] - 206s 1s/step - loss: 0.3100 - accuracy: 0.8466 - dice_coef: 0.6901 - val_loss: 0.3106 - val_accuracy: 0.8300 - val_dice_coef: 0.6893
Epoch 39/50
156/156 [==============================] - 205s 1s/step - loss: 0.3178 - accuracy: 0.8432 - dice_coef: 0.6820 - val_loss: 0.2965 - val_accuracy: 0.8651 - val_dice_coef: 0.7037
Epoch 40/50
156/156 [==============================] - 205s 1s/step - loss: 0.3144 - accuracy: 0.8444 - dice_coef: 0.6850 - val_loss: 0.3026 - val_accuracy: 0.8430 - val_dice_coef: 0.6972
Epoch 41/50
156/156 [==============================] - 206s 1s/step - loss: 0.3108 - accuracy: 0.8468 - dice_coef: 0.6890 - val_loss: 0.3420 - val_accuracy: 0.7855 - val_dice_coef: 0.6577
Epoch 42/50
156/156 [==============================] - 205s 1s/step - loss: 0.3128 - accuracy: 0.8473 - dice_coef: 0.6874 - val_loss: 0.2952 - val_accuracy: 0.8671 - val_dice_coef: 0.7049
Epoch 43/50
156/156 [==============================] - 205s 1s/step - loss: 0.3090 - accuracy: 0.8477 - dice_coef: 0.6909 - val_loss: 0.2951 - val_accuracy: 0.8618 - val_dice_coef: 0.7048
Epoch 44/50
156/156 [==============================] - 206s 1s/step - loss: 0.3065 - accuracy: 0.8497 - dice_coef: 0.6937 - val_loss: 0.2993 - val_accuracy: 0.8353 - val_dice_coef: 0.7008
Epoch 45/50
156/156 [==============================] - 205s 1s/step - loss: 0.3056 - accuracy: 0.8493 - dice_coef: 0.6944 - val_loss: 0.2918 - val_accuracy: 0.8498 - val_dice_coef: 0.7081
Epoch 46/50
156/156 [==============================] - 206s 1s/step - loss: 0.3041 - accuracy: 0.8494 - dice_coef: 0.6960 - val_loss: 0.3002 - val_accuracy: 0.8376 - val_dice_coef: 0.6996
Epoch 47/50
156/156 [==============================] - 206s 1s/step - loss: 0.3055 - accuracy: 0.8499 - dice_coef: 0.6945 - val_loss: 0.2944 - val_accuracy: 0.8518 - val_dice_coef: 0.7057
Epoch 48/50
156/156 [==============================] - 206s 1s/step - loss: 0.3073 - accuracy: 0.8477 - dice_coef: 0.6922 - val_loss: 0.2822 - val_accuracy: 0.8669 - val_dice_coef: 0.7178
Epoch 49/50
156/156 [==============================] - 206s 1s/step - loss: 0.3057 - accuracy: 0.8501 - dice_coef: 0.6945 - val_loss: 0.2898 - val_accuracy: 0.8716 - val_dice_coef: 0.7104
Epoch 50/50
156/156 [==============================] - 206s 1s/step - loss: 0.3000 - accuracy: 0.8503 - dice_coef: 0.7001 - val_loss: 0.2878 - val_accuracy: 0.8623 - val_dice_coef: 0.7120

```

![UNET_reusult.jpg](attachment:UNET_reusult.jpg)

![evaluate_test.jpg](attachment:evaluate_test.jpg)


```python

```
