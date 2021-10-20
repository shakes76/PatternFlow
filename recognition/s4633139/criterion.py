#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021, H.WAKAYAMA, All rights reserved.
#  File: criterion.py
#  Author: Hideki WAKAYAMA
#  Contact: h.wakayama@uq.net.au
#  Platform: macOS Big Sur Ver 11.2.1, Pycharm pro 2021.1
#  Time: 20/10/2021, 09:52
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#dice coefficient
def dice_coef(pred, target):
  """
  function to compute the dice coefficient
  param----
    pred(tensor[B,C,W,H]): predicted mask images
    target(tensor[B,C,W,H]: target mask images
  return---
    dice coefficient
  """
  batch_size = len(pred)
  somooth = 1.

  pred_flat = pred.view(batch_size, -1)
  target_flat = target.view(batch_size, -1)

  intersection = (pred_flat*target_flat).sum()
  dice_coef = (2.*intersection+somooth)/(pred_flat.sum()+target_flat.sum()+somooth)
  return dice_coef


#loss
def dice_loss(pred, target):
  """
  function to compute dice loss
  param----
    pred(tensor[B,C,W,H]): predicted mask images
    target(tensor[B,C,W,H]): target mask images
  return----
    dice loss
  """
  dice_loss = 1 - dice_coef(pred, target)
  return dice_loss