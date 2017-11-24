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

    return 

def loop(model, epoch=30, summary_dir='./summary/test'):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    logging.info('session initialized')
    
    setup_summary_dir(summary_dir)
    writer = tf.summary.FileWriter(summary_dir, sess.graph)
    
    history = []
    for e in range(epoch):
        result = {
            'train':train(sess, model),
            'valid':valid(sess, model)
        }
        history.append( result )
        
        message = ['[{:04d}]'.format(e)]
        for key, values in result.items():
            message.append( '[{}] cost:{:.3f} accuracy:{:.3f} elapsed:{:.3f}sec'.format(key, values['cost'], values['accuracy'], values['elapsed']) )
        logging.info(' '.join( message ))
        writer.add_summary(result['train']['summary'], result['train']['step'])
    return history

def train(sess, model):
    model.dataflow['train'].reset_state()
    costs, accuracies = [], []
    
    elapsed = 0
    for datapoint in model.dataflow['train'].get_data():
        timestamp = time.time()
        datapoint.append( sklearn.preprocessing.label_binarize( datapoint[1], range(10) ).astype(np.float32) )
        _, cost, accuracy = sess.run(
            [model.op, model.cost, model.accuracy],
            feed_dict=dict(zip(model.inputs, datapoint))
#            options=run_options,
#            run_metadata=run_metadata
        )
        elapsed += time.time() - timestamp

        costs.append(cost)
        accuracies.append(accuracy)
        
        summary_str, step = sess.run([model.summary_merged, model.global_step], feed_dict=dict(zip(model.inputs, datapoint)) )
    return {'cost':np.mean(costs), 'accuracy':np.mean(accuracies), 'elapsed':elapsed, 'summary':summary_str, 'step':step}

def valid(sess, model):
    model.dataflow['valid'].reset_state()
    costs, accuracies = [], []
    
    elapsed = 0
    for datapoint in model.dataflow['valid'].get_data():
        timestamp = time.time()
        datapoint.append( sklearn.preprocessing.label_binarize( datapoint[1], range(10) ).astype(np.float32) )
        cost, accuracy = sess.run(
            [model.cost, model.accuracy],
            feed_dict=dict(zip(model.inputs, datapoint)))
        elapsed += time.time() - timestamp
        
        costs.append(cost)
        accuracies.append(accuracy)
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
        ax.legend()
    plt.tight_layout()
    