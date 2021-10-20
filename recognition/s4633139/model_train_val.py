#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021, H.WAKAYAMA, All rights reserved.
#  File: model_train_val.py
#  Author: Hideki WAKAYAMA
#  Contact: h.wakayama@uq.net.au
#  Platform: macOS Big Sur Ver 11.2.1, Pycharm pro 2021.1
#  Time: 20/10/2021, 09:52
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from criterion import dice_coef, dice_loss
from tqdm import tqdm
import torch

def model_train_val(model, optimizer, EPOCHS, train_loader, val_loader):
  """
  function for model training and validation
  :param----
    model: model
    optimizer: optimizer
    EPOCHS(int):number of epochs
    train_loader: train loader
    val_loader: validation loader
  :return----
    list of train and validation dice coefficients and dice losses by epochs
  """

  TRAIN_LOSS = []
  TRAIN_DICE = []
  VAL_LOSS =[]
  VAL_DICE = []
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  for epoch in range(1, EPOCHS+1):
    print('EPOCH {}/{}'.format(epoch, EPOCHS))
    running_loss = 0
    running_dicecoef = 0
    running_loss_val = 0
    running_dicecoef_val = 0
    BATCH_NUM = len(train_loader)
    BATCH_NUM_VAL = len(val_loader)

    #train
    with tqdm(train_loader, unit='batch') as tbatch:
      for batch_idx, (x, y) in enumerate(tbatch):
        x, y = x.to(device), y.to(device)
        tbatch.set_description(f'Batch: {batch_idx}')

        optimizer.zero_grad()
        output = model(x)
        loss = dice_loss(output, y)
        dicecoef = dice_coef(output, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_dicecoef += dicecoef.item()

        tbatch.set_postfix(loss=loss.item(), dice_coef=dicecoef.item())

    epoch_loss = running_loss/BATCH_NUM
    epoch_dicecoef = running_dicecoef/BATCH_NUM
    TRAIN_LOSS.append(epoch_loss)
    TRAIN_DICE.append(epoch_dicecoef)

    #validation
    with tqdm(val_loader, unit='batch') as valbatch:
      for batch_idx, (x, y) in enumerate(valbatch):
        x, y = x.to(device), y.to(device)
        valbatch.set_description(f'Batch: {batch_idx}')
        output_val = model(x)
        loss_val = dice_loss(output_val, y)
        dicecoef_val = dice_coef(output_val, y)
        valbatch.set_postfix(loss=loss_val.item(), dice_coef=dicecoef_val.item())

        running_loss_val += loss_val.item()
        running_dicecoef_val += dicecoef_val.item()

    VAL_LOSS.append(running_loss_val/BATCH_NUM_VAL)
    VAL_DICE.append(running_dicecoef_val/BATCH_NUM_VAL)

  return TRAIN_DICE, VAL_DICE, TRAIN_LOSS, VAL_LOSS