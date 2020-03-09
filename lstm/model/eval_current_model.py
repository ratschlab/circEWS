import csv
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf

from time import time
from os import mkdir, listdir
from os.path import join, exists
from sys import stdout
from dataloader import RawSignals, TopFeatures
from utils import tf_proba, eval_prob


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dim_hidden_dynamic', type=int, default=2000)
parser.add_argument('--dim_hidden_static', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--use_standard_scaler', type=int, default=1)
parser.add_argument('--use_weight', type=int, default=0)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--max_wait_epochs', type=int, default=10)

parser.add_argument('--input_type', default='feature', choices=['feature', 'signal'])
parser.add_argument('--topk', type=int, default=500)
parser.add_argument('--topk_type', default='feature', choices=['feature', 'variable'])

parser.add_argument('--data_split', default='temporal_5')
parser.add_argument('--early_stop_by', choices=['auprc', 'objective'], default='auprc')
parser.add_argument('--test_mode', action='store_true')

args = parser.parse_args()
dim_hidden_dynamic = args.dim_hidden_dynamic
dim_hidden_static = args.dim_hidden_static
learning_rate = args.learning_rate
batch_size = args.batch_size
dropout = args.dropout
use_standard_scaler = bool(args.use_standard_scaler)
use_weight = bool(args.use_weight)
max_epochs = args.max_epochs
max_wait_epochs = args.max_wait_epochs

input_type = args.input_type
topk = args.topk
topk_type = args.topk_type
data_split = args.data_split
early_stop_by = args.early_stop_by
test_mode = args.test_mode


# dim_hidden_dynamic = 440
# dim_hidden_static = 9
# learning_rate = 0.0063
# batch_size = 6
# dropout = 0.24
# use_weight = False
# use_standard_scaler = True

# input_type = 'feature'
# topk = 500
# topk_type = 'feature'
# max_epochs = 100
# max_wait_epochs = 5
# data_split = 'temporal_5'
# early_stop_by = 'auprc'
# test_mode = True


scaler_type = 'standard_scaled' if use_standard_scaler else 'minmax_scaled'
keep_prob = 1 - dropout


data_version = '180918'

res_dir = 'me-%d_lr-%g_dh-%d-%d_dp-%g_bs-%d_w-%d_%s'%(max_epochs,
                                                      learning_rate, 
                                                      dim_hidden_dynamic, 
                                                      dim_hidden_static, 
                                                      dropout,
                                                      batch_size,
                                                      int(use_weight),
                                                      scaler_type.split('_')[0])

if input_type == 'signal':
    dataloader = RawSignals(scaler_type=scaler_type,
                            data_split=data_split, 
                            data_version=data_version,
                            test_mode=test_mode)
    res_path = join(dataloader.bern_path, '8_predictions', 'lstm_raw', data_version, data_split, 'all_vars', res_dir)
else:
    dataloader = TopFeatures(topk=topk,
                             topk_type=topk_type,
                             scaler_type=scaler_type,
                             data_split=data_split, 
                             data_version=data_version,
                             test_mode=test_mode)
    res_path = join(dataloader.bern_path, '8_predictions', 'lstm', 'v6b', 'reduced', data_split, 
                    'top%d_%s'%(topk, 'features' if topk_type=='feature' else 'vars'), res_dir)


if not exists(res_path):
    print('Model not trained.')
    exit(0)


if exists(join(res_path, 'checkpoint')):
    save_path = open(join(res_path, 'checkpoint'), 'r').readlines()[-1].split('"')[1]
else:
    print('Model not learned.')
    exit(1)


X, y, s, dt, info = dataloader.load_test_data()
    

dim_input_dynamic = X[0].shape[1]
dim_output = y[0].shape[1]
dim_input_static = len(s[0])


graph = tf.Graph()
with graph.as_default():
    # for dynamic variables
    lstm = tf.contrib.rnn.LSTMCell(dim_hidden_dynamic)
    W_dynamic = tf.get_variable('W_dynamic', shape=[dim_hidden_dynamic, dim_output])

    # for static variables
    W_static_0 = tf.get_variable('W_static_0', shape=[dim_input_static, dim_hidden_static])
    b_static_0 = tf.get_variable('b_static_0', shape=[dim_hidden_static])

    W_static_1 = tf.get_variable('W_static_1', shape=[dim_hidden_static, dim_output])

    # for both static and dynamic variables
    b = tf.get_variable('b', shape=[dim_output], initializer=tf.zeros_initializer)

    # hp placeholder
    keep_prob_ = tf.placeholder(tf.float32, shape=[], name='Keep_Prob')


    # data placeholder
    D_ = tf.placeholder(tf.float32, shape=[None, None, dim_input_dynamic], name='Input_Dynamic')
    S_ = tf.placeholder(tf.float32, shape=[None, dim_input_static], name='Input_Static')
    seq_len_ = tf.placeholder(tf.float32, shape=[None], name='Sequence_Length')
    
    tf_holder = dict(D_=D_, S_=S_, seq_len_=seq_len_, keep_prob_=keep_prob_)
    tf_var = dict(lstm=lstm, Wd=W_dynamic, Ws0=W_static_0, bs0=b_static_0, Ws1=W_static_1, b=b)

    _, proba = tf_proba(tf_var, tf_holder)
    
    saver = tf.train.Saver(max_to_keep=5)
    sess =  tf.Session(graph=graph)
    saver.restore(sess, save_path)
    stdout.write('Restored.\n')
    stdout.flush()
    
    data = dict(X=X, y=y, s=s)
    all_prob = eval_prob(sess, proba, data, tf_holder, batch_size, 'test', res_path)

    if not exists(join(res_path, 'proba')):
        mkdir(join(res_path, 'proba'))

    # save test scores
    t = time()
    for i in range(len(X)):
        df = pd.DataFrame(np.hstack((y[i], all_prob[i])), 
                          columns=['TrueLabel', 'PredScore'])
        df['AbsDatetime'] = dt[i]
        df['AbsDatetime'] = pd.to_datetime(df.AbsDatetime, unit='ns')
        df.set_index('AbsDatetime', drop=True, inplace=True)
        pid, f = info[i]
        df.to_hdf(join(res_path, 'proba', f), pid)
        if (i+1)%10 == 0:
            stdout.write('%d patients\' scores have been written to disk. Time:%4.4g sec.\n'%(i+1, time()-t))
            stdout.flush()

    sess.close()

