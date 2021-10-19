#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021, H.WAKAYAMA, All rights reserved.
#  File: IUNet_criterion.py
#  Author: Hideki WAKAYAMA
#  Contact: h.wakayama@uq.net.au
#  Platform: macOS Big Sur Ver 11.2.1, Pycharm pro 2021.1
#  Time: 19/10/2021, 15:47
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#dice coefficient
def dice_coef(pred, target):
  batch_size = len(pred)
  somooth = 1.

  pred_flat = pred.view(batch_size, -1)
  target_flat = target.view(batch_size, -1)

  intersection = (pred_flat*target_flat).sum()
  dice_coef = (2.*intersection+somooth)/(pred_flat.sum()+target_flat.sum()+somooth)
  return dice_coef


#loss
def dice_loss(pred, target):
  dice_loss = 1 - dice_coef(pred, target)
  return dice_loss