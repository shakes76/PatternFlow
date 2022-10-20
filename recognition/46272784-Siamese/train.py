# This file contains the source code for training, validating, testing and saving my model
import os
import sys
sys.path.insert(1, os.getcwd())
import modules
from dataset import loadFile
import tensorflow as tf
from tensorflow import keras
import csv

def saveOption(optimizer, siamese):
    """
    Setup the checkpoint directory
    """
    checkpoint_dir = os.path.join(os.getcwd(), "Siamese_ckeckpoint")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                  net=siamese)
    return checkpoint_prefix, checkpoint

@tf.function
def train_step(pairs, optimizer, siamese, train_acc_metric):
    """
    Training process: compute loss, apply gradient
    """
    with tf.GradientTape() as gra_tape:
        y_true = pairs[2]
        # print(pairs[0], pairs[1])
        y_pred = siamese([pairs[0], pairs[1]], training=True)
        lossValue = (modules.loss())(y_true, y_pred)
        
    gradient = gra_tape.gradient(lossValue, siamese.trainable_weights)
    optimizer.apply_gradients(zip(gradient, siamese.trainable_weights))
    train_acc_metric.update_state(y_true, y_pred)
    
    return lossValue

@tf.function
def valid_step(pairs, siamese, test_acc_metric):
    """
    Compute validation loss
    """
    y_true = pairs[2]
    y_pred = siamese([pairs[0], pairs[1]], training=False)
    lossValue = (modules.loss())(y_true, y_pred)
    test_acc_metric.update_state(y_true, y_pred)
    return lossValue

@tf.function
def test_step(pairs, siamese, acc_metric):
    """
    Compute testing loss
    """
    y_true = pairs[2]
    y_pred = siamese([pairs[0], pairs[1]], training=False)
    lossValue = (modules.loss())(y_true, y_pred)
    acc_metric.update_state(y_true, y_pred)
    return lossValue

def train(train_ds, valid_ds, test_ds, epochs, train_step, checkpoint_prefix, checkpoint, optimizer, siamese):
    """
    Training the siamese network.
    Returns a dictionary of train/validation loss and accuracy
    """
    # print(train_ds)
    info = {'train_loss': [],
            'train_accu': [],
            'valid_loss': [],
            'valid_accu': []}
    loss_tracker = keras.metrics.Mean()
    acc_metric = keras.metrics.BinaryAccuracy()
    for epoch in range(epochs):
        print('>>>>>>>>> Epoch {}'.format(epoch+1))
        # count = 0
        # siameseLoss = 0
        batchnum = 0
        # train
        for batch in train_ds:
            if batchnum % 100 == 1:
                print('>> Training batch {}'.format(batchnum))
                print('> Train_loss {}'.format(loss_tracker.result().numpy()))
                print('> Train_accu {}'.format(acc_metric.result().numpy()))
            lossValue = train_step(batch, optimizer, siamese, acc_metric)
            loss_tracker.update_state(lossValue)
            # siameseLoss += lossValue
            # count += 1
            batchnum += 1
        info['train_loss'].append(loss_tracker.result().numpy())
        train_accu = acc_metric.result().numpy()
        info['train_accu'].append(train_accu)
        acc_metric.reset_states()
        loss_tracker.reset_states()
        
        # v_count = 0
        # v_siameseLoss = 0
        batchnum = 0
        # validate
        for batch in valid_ds:
            if batchnum % 100 == 1:
                print('>> Validating batch {}'.format(batchnum))
                print('> Valid_loss {}'.format(loss_tracker.result().numpy()))
            lossValue = valid_step(batch, siamese, acc_metric)
            loss_tracker.update_state(lossValue)
            # v_siameseLoss += lossValue
            # v_count += 1
            batchnum += 1
           
        info['valid_loss'].append(loss_tracker.result().numpy())   
        valid_accu = acc_metric.result().numpy()
        info['valid_accu'].append(valid_accu)     
        acc_metric.reset_states()
        loss_tracker.reset_states()
        
        # Save the model every epochs
        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)   
                
        row = [epoch, (info['train_loss'])[-1], (info['train_accu'])[-1], (info['valid_loss'])[-1], (info['valid_accu'])[-1]]
        print('Epoch {} | Train loss {} | Train Accu {} | Valid loss {} | Valid Accu {}'.
              format(row[0], row[1], row[2], row[3], row[4]))
        f = open('./record.csv', 'a+')
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()
        
    # Test
    for batch in test_ds:
        _ = test_step(batch, siamese, acc_metric)
    print('Test accu: {}'.format(acc_metric.result().numpy()))
    acc_metric.reset_states()
    
    # Save the model
    siamese.save("siamese.h5")
    return info

def main():
    tr_a, tr_n, v_a, v_n, te_a, te_n = loadFile('F:/AI/COMP3710/data/AD_NC/')
    train_ds = modules.generatePairs(tr_a, tr_n)
    valid_ds = modules.generatePairs(v_a, v_n)
    test_ds = modules.generatePairs(te_a, te_n)
    opt = keras.optimizers.Adam(1e-4)
    cnn = modules.makeCNN()
    siamese = modules.makeSiamese(cnn)
    checkpoint_prefix, checkpoint = saveOption(opt, siamese)
    checkpoint.restore(tf.train.latest_checkpoint(r'F:\AI\COMP3710\PatternFlow\recognition\46272784-Siamese\Siamese_ckeckpoint'))
    history = train(train_ds, valid_ds, test_ds, 10, train_step, checkpoint_prefix, checkpoint, opt, siamese)
    cnn.save('cnn.h5')
    
    # # results = siamese.evaluate(vd)
    # # print("test loss, test acc:", results)
    
if __name__ == "__main__":
    main()