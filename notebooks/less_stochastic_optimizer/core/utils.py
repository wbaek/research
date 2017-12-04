import os
import shutil
import time

import numpy as np
import tensorflow as tf
import sklearn.preprocessing

import logging
logger = logging.getLogger(__name__)

def setup_summary_dir(summary_dir = './summary/test'):
    if os.path.exists(summary_dir):
        shutil.rmtree(summary_dir)
    os.makedirs(summary_dir)

def loop(sess, model, epoch=30, summary_dir='./summary/test', epoch_steps=1e10):
    sess.run(tf.global_variables_initializer())
    logger.info('session initialized')
    
    setup_summary_dir(summary_dir)
    train_writer = tf.summary.FileWriter(summary_dir+'/train', sess.graph)
    valid_writer = tf.summary.FileWriter(summary_dir+'/valid')
    
    history = []
    for e in range(epoch):
        result = {
            'train':train(sess, model, train_writer, epoch_steps),
            'valid':valid(sess, model, valid_writer)
        }
        history.append( result )
        
        message = ['[{:04d}]'.format(e)]
        for key in ['train', 'valid']:
            values = result[key]
            message.append( '[{}] cost:{:.3f} accuracy:{:.3f} elapsed:{:.3f}sec'.format(key, values['cost'], values['accuracy'], values['elapsed']) )
        logger.info(' '.join( message ))
    sess.close()
    return history

def train(sess, model, writer, steps=1e10):
    model.dataflow['train'].reset_state()
    costs, accuracies = [], []
    
    elapsed = 0
    s = 0
    for datapoint in model.dataflow['train'].get_data():
        s += 1
        if s > steps:
            break
        
        timestamp = time.time()
        datapoint.append( sklearn.preprocessing.label_binarize( datapoint[1], range(10) ).astype(np.float32) )
        _, cost, accuracy = sess.run(
            [model.op, model.cost, model.accuracy],
            feed_dict=dict(zip(model.inputs, datapoint))
#            options=run_options,
#            run_metadata=run_metadata
        )
        e = time.time() - timestamp
        elapsed += e
        logger.debug('[train] cost:{:.3f} accuracy:{:.3f} elapsed:{:.3f}sec'.format(cost, accuracy, e) )

        costs.append(cost)
        accuracies.append(accuracy)
        
        summary_str, step = sess.run([model.summary_merged, model.global_step], feed_dict=dict(zip(model.inputs, datapoint)) )
        writer.add_summary(summary_str, (step+1)*model.batch_size)
    return {'cost':np.mean(costs), 'accuracy':np.mean(accuracies), 'elapsed':elapsed}

def valid(sess, model, writer):
    model.dataflow['valid'].reset_state()
    costs, accuracies = [], []
    
    elapsed = 0
    for datapoint in model.dataflow['valid'].get_data():
        timestamp = time.time()
        datapoint.append( sklearn.preprocessing.label_binarize( datapoint[1], range(10) ).astype(np.float32) )
        cost, accuracy = sess.run(
            [model.cost, model.accuracy],
            feed_dict=dict(zip(model.inputs, datapoint)))
        e = time.time() - timestamp
        elapsed += e
        logger.debug('[valid] cost:{:.3f} accuracy:{:.3f} elapsed:{:.3f}sec'.format(cost, accuracy, e) )
        
        costs.append(cost)
        accuracies.append(accuracy)
        
        summary_str, step = sess.run([model.summary_merged, model.global_step], feed_dict=dict(zip(model.inputs, datapoint)) )
        writer.add_summary(summary_str, (step+1)*model.batch_size)
    return {'cost':np.mean(costs), 'accuracy':np.mean(accuracies), 'elapsed':elapsed}

def plot_jupyter(history):
    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4*3))
    for i, label in enumerate(['cost', 'accuracy']):
        ax = fig.add_subplot(3, 1, i+1)
        ax.set_xlabel('epoch', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=12)
        for j, mode in enumerate(['train', 'valid']):
            if not label in history[0][mode]:
                continue
            ax.plot(np.array([h for h in range(len(history))]), np.array([np.sum(h[mode][label]) for h in history]), label=mode)
        ax.set_ylim([0.0,1.0])
        ax.legend()
    plt.tight_layout()
    
    print('average elapsed time train:{:.6f}sec valid:{:.6f}sec'.format(
        np.mean([h['train']['elapsed'] for h in history[5:]]),
        np.mean([h['valid']['elapsed'] for h in history[5:]]) )
         )
    