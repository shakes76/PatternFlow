#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021, H.WAKAYAMA, All rights reserved.
#  File: IUNet_train_test.py
#  Author: Hideki WAKAYAMA
#  Contact: h.wakayama@uq.net.au
#  Platform: macOS Big Sur Ver 11.2.1, Pycharm pro 2021.1
#  Time: 19/10/2021, 15:47
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from IUNet_criterion import dice_coef, dice_loss
from tqdm import tqdm

def model_train_test(model, optimizer, EPOCHS, train_loader, test_loader):
  """function for model training and test
  :return: list of train and test dice coefficients and dice losses by epochs
  """
  TRAIN_LOSS = []
  TRAIN_DICE = []
  TEST_LOSS =[]
  TEST_DICE = []

  for epoch in range(1, EPOCHS+1):
    print('EPOCH {}/{}'.format(epoch, EPOCHS))
    running_loss = 0
    running_dicecoef = 0
    running_loss_test = 0
    running_dicecoef_test = 0
    BATCH_NUM = len(train_loader)
    BATCH_NUM_TEST = len(test_loader)

    #train
    with tqdm(train_loader, unit='batch') as tbatch:
      for batch_idx, (x, y) in enumerate(tbatch):
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

    #test
    with tqdm(test_loader, unit='batch') as tsbatch:
      for batch_idx, (x, y) in enumerate(tsbatch):
        tsbatch.set_description(f'Batch: {batch_idx}')
        output_test = model(x)
        loss_test = dice_loss(output_test, y)
        dicecoef_test = dice_coef(output_test, y)
        tsbatch.set_postfix(loss=loss_test.item(), dice_coef=dicecoef_test.item())

        running_loss_test += loss_test.item()
        running_dicecoef_test += dicecoef_test.item()

    TEST_LOSS.append(running_loss_test/BATCH_NUM_TEST)
    TEST_DICE.append(running_dicecoef_test/BATCH_NUM_TEST)

  return TRAIN_DICE, TEST_DICE, TRAIN_LOSS, TEST_LOSS