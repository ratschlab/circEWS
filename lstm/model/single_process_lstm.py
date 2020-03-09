import h5py
import csv
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
import tensorflow as tf

from time import time
from os import makedirs, mkdir, listdir
from os.path import join, exists
from sys import stdout
from dataloader import RawSignals, TopFeatures
from utils import sequence2matrix, tf_proba, eval_prob, eval_loss_and_prob


import argparse
parser = argparse.ArgumentParser()
# hyper-parameters
parser.add_argument('--dim_hidden_dynamic', type=int, default=500)
parser.add_argument('--dim_hidden_static', type=int, default=10)
parser.add_argument('--dim_hidden_joint', type=int, default=128)
parser.add_argument('--activ_func', default='tanh', choices=['tanh', 'relu', 'sigmoid'])
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--scaler_type', default='std', choices=['std', 'minmax'])
parser.add_argument('--use_weight', action='store_true')

# data configurations
parser.add_argument('--input_type', default='feature', choices=['feature', 'signal'])
parser.add_argument('--topk_type', default='feature', choices=['feature', 'variable'])
parser.add_argument('--topk', type=int, default=500)
parser.add_argument('--data_split', default='temporal_5')

# other configurations
parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--max_wait_epochs', type=int, default=5)
parser.add_argument('--early_stop_by', default='auprc', choices=['auprc', 'objective'])
parser.add_argument('--test_mode', action='store_true')


args = parser.parse_args()
dim_hidden_dynamic = args.dim_hidden_dynamic
dim_hidden_static = args.dim_hidden_static
dim_hidden_joint = args.dim_hidden_joint
activ_func = args.activ_func
learning_rate = args.learning_rate
batch_size = args.batch_size
dropout = args.dropout
scaler_type = args.scaler_type
use_weight = args.use_weight
max_epochs = args.max_epochs
max_wait_epochs = args.max_wait_epochs

input_type = args.input_type
topk = args.topk
topk_type = args.topk_type
data_split = args.data_split
early_stop_by = args.early_stop_by
test_mode = args.test_mode


# dim_hidden_dynamic = 70
# dim_hidden_static = 4
# dim_hidden_joint = 50
# activ_func = 'relu'
# learning_rate = 0.01
# batch_size = 4
# dropout = 0.5
# use_weight = True
# scaler_type = 'standard'

# input_type = 'signal'                                                                                                                                                                                                        
# topk = 500                                                                                                                                                                                                                    
# topk_type = 'feature' 
# max_epochs = 2
# max_wait_epochs = 5
# data_split = 'temporal_5'
# early_stop_by = 'auprc'
# test_mode = True



data_version = '180918'
keep_prob = 1 - dropout



if input_type == 'signal':
    dataloader = RawSignals(scaler_type=scaler_type,
                            data_split=data_split, 
                            data_version=data_version,
                            ngpus=1,
                            test_mode=test_mode)
else:
    dataloader = TopFeatures(topk=topk,
                             topk_type=topk_type,
                             scaler_type=scaler_type,
                             data_split=data_split, 
                             data_version=data_version,
                             ngpus=1,
                             test_mode=test_mode)
    

res_dir = 'me-%d_lr-%g_dh-%d-%d-%d_f-%s_dp-%g_bs-%d_w-%d_%s'%(max_epochs,
                                                              learning_rate, 
                                                              dim_hidden_dynamic, 
                                                              dim_hidden_static, 
                                                              dim_hidden_joint,
                                                              activ_func, 
                                                              dropout,
                                                              batch_size,
                                                              int(use_weight),
                                                              scaler_type)

res_tmp = input_type if input_type=='signal' else 'top%d_%s'%(topk, 'features' if topk_type=='feature' else 'vars')
res_path = join(dataloader.bern_path, '8_predictions', 'lstm', 'v6b', 'reduced', data_split, res_tmp, res_dir)


if not exists(res_path):
    stdout.write('Not exists\n')
    makedirs(res_path)


last_saved_epoch = None
if exists(join(res_path, 'checkpoint')):
    save_path = open(join(res_path, 'checkpoint'), 'r').readlines()[-1].split('"')[1]
    last_trained_epoch = pd.read_csv(join(res_path, 'training_info.csv')).epoch.max()
    last_saved_epoch = int(save_path.split('-')[-1])
    
    if last_trained_epoch==max_epochs or last_trained_epoch-last_saved_epoch==max_wait_epochs:
        stdout.write('Model learned.\n')
    print(res_path)

# train_X, train_y, train_s = dataloader.load_train_data(0)

val_X, val_y, val_s = dataloader.load_val_data()
data=dict(X=val_X, y=val_y, s=val_s)

dim_input_dynamic = val_X[0].shape[1]
dim_output = val_y[0].shape[1]
dim_input_static = len(val_s[0])


if use_weight:
    vec_train_label = np.vstack(tuple(train_y))
    pos_weight = np.sum(vec_train_label<2) / np.sum(vec_train_label==1) - 1
    scale = np.sum(vec_train_label<2) / ( 2 * np.sum(vec_train_label==0) )

if test_mode:
    num_batches_per_epoch = 20 / batch_size
else:
    num_batches_per_epoch = int(np.round(len(dataloader.train_pids_splits[0]) / batch_size))


graph = tf.Graph()
with graph.as_default():
    # for dynamic variables
    lstm = tf.contrib.rnn.LSTMCell(dim_hidden_dynamic)

    # for static variables
    W_static_0 = tf.get_variable('W_static_0', shape=[dim_input_static, dim_hidden_static])
    b_static_0 = tf.get_variable('b_static_0', shape=[dim_hidden_static])

    W_joint = tf.get_variable('W_joint', shape=[dim_hidden_dynamic+dim_hidden_static, dim_hidden_joint])
    
    # for both static and dynamic variables
    b_joint = tf.get_variable('b_joint', shape=[dim_hidden_joint], initializer=tf.zeros_initializer)
    
    W_out = tf.get_variable('W_out', shape=[dim_hidden_joint, dim_output])
    b_out = tf.get_variable('b_out', shape=[dim_output], initializer=tf.zeros_initializer)
    
    # hp placeholders
    keep_prob_ = tf.placeholder(tf.float32, shape=[], name='Keep_Prob')


    # data placeholders
    D_ = tf.placeholder(tf.float32, shape=[None, None, dim_input_dynamic], name='Input_Dynamic')
    S_ = tf.placeholder(tf.float32, shape=[None, dim_input_static], name='Input_Static')
    y_ = tf.placeholder(tf.float32, shape=[None, None, dim_output], name='Output')
    mask = tf.placeholder(tf.bool, shape=[None, None], name='Mask')
    seq_len_ = tf.placeholder(tf.float32, shape=[None], name='Sequence_Length')

    # for updating the model
    global_step = tf.get_variable('global_step', 
                                  shape=[], 
                                  initializer=tf.zeros_initializer, 
                                  trainable=False, 
                                  dtype=tf.int32)

    tf_holder = dict(D_=D_, S_=S_, y_=y_, mask=mask, seq_len_=seq_len_, keep_prob_=keep_prob_)
    tf_var = dict(lstm=lstm, Ws0=W_static_0, bs0=b_static_0, Wj=W_joint, bj=b_joint,
                  Wo=W_out, bo=b_out)
    output, proba = tf_proba(tf_var, tf_holder, activ_func)
    
    masked_y = tf.boolean_mask(y_, mask)
    masked_o = tf.boolean_mask(output, mask)
    if use_weight:
        loss = scale*tf.nn.weighted_cross_entropy_with_logits(masked_y, masked_o, pos_weight)
    else:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=masked_y, logits=masked_o)
    loss = tf.reduce_mean(loss)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()



    saver = tf.train.Saver(max_to_keep=5)
    sess =  tf.Session(graph=graph)
    if exists(join(res_path, 'checkpoint')):
        saver.restore(sess, save_path)
        stdout.write('Restored\n')
    else:
        sess.run(init)
        stdout.write('Initialized\n')

    cnt_epoch = 0
    cnt_wait_epoch = 0
    cnt_iter = 0
    cnt_train = 0
    cnt_split = 0

    train_t = 0
    epoch_t = 0
    
    min_loss = float('Inf')
    max_auprc = 0

    if last_saved_epoch is None:
        with open(join(res_path, 'training_info.csv'), 'w+') as f:
            line = ['epoch', 'train_loss', 'val_loss', 'val_auroc', 'val_auprc', 'val_prev', 'train_time', 'epoch_time']
            writer = csv.writer(f)
            writer.writerow(line)
            stdout.write(' '.join(line)+'\n')
    elif last_saved_epoch > 0:
        train_info = pd.read_csv(join(res_path, 'training_info.csv'))
        train_info = train_info[train_info.epoch<=last_saved_epoch]
        if early_stop_by == 'objective':
            min_loss = train_info.iloc[-1].val_loss
        else:
            max_auprc = train_info.iloc[-1].val_auprc
        
        train_info.to_csv(join(res_path, 'training_info.csv'), index=False)
        cnt_epoch = last_saved_epoch
        cnt_iter = last_saved_epoch * num_batches_per_epoch

    if last_saved_epoch is None:
        with open(join(res_path, 'training_log.csv'), 'w+') as f:
            line = ['iter', 'batch_loss']
            writer = csv.writer(f)
            writer.writerow(line)
    else:
        train_log = pd.read_csv(join(res_path, 'training_log.csv'))
        train_log = train_log[train_log['iter']<=cnt_iter]
        train_log.to_csv(join(res_path, 'training_log%d.csv'), index=False)
        

    stop_training = False
    reset = True


    permuted_train_indices = np.random.permutation(len(dataloader.train_pids_splits[0]))
    while not stop_training and cnt_epoch < max_epochs:
    # while not stop_training:
        if reset:
            reset = False
            train_loss = 0
            total_cnt = 0
            t = time()
            
        batch_train_indices = permuted_train_indices[cnt_train:min(cnt_train+batch_size, len(dataloader.train_pids_splits[0]))]
        
        cnt_train += batch_size
        if cnt_train > len(dataloader.train_pids_splits[0]):
            permuted_train_indices = np.random.permutation(permuted_train_indices)
            batch_train_indices = np.concatenate((batch_train_indices,
                                                  permuted_train_indices[:cnt_train-len(dataloader.train_pids_splits[0])]))
            cnt_train -= len(dataloader.train_pids_splits[0])            

        batch_dynamic, batch_output, batch_static = dataloader.load_train_data_batch(0, batch_size)


        mat_dynamic, mat_output, mat_mask, seq_len = sequence2matrix(batch_dynamic, batch_output, batch_size)
        if np.sum(mat_mask.sum(axis=1) == 0) > 0:
            idx_keep = np.where(mat_mask.sum(axis=1)>0)[0]
            mat_dynamic = mat_dynamic[idx_keep]
            mat_output = mat_output[idx_keep]
            mat_mask = mat_mask[idx_keep]
            seq_len = np.array(seq_len)[idx_keep]
            batch_static = batch_static[idx_keep]

        if len(mat_dynamic) > 0:
            _, batch_loss = sess.run([train_op, loss], feed_dict={D_: mat_dynamic, 
                                                                   y_: mat_output, 
                                                                   mask: mat_mask,
                                                                   S_: batch_static,
                                                                   seq_len_: seq_len,
                                                                   keep_prob_: keep_prob})
            train_loss += batch_loss * len(batch_dynamic)
            total_cnt += len(batch_dynamic)
        cnt_iter += 1


        if cnt_iter%10 == 0:    
            with open(join(res_path, 'training_log.csv'), 'a') as f:
                line = [cnt_iter, batch_loss]
                writer = csv.writer(f)
                writer.writerow(line)
                stdout.write(' '.join(['%4.4g'%tmp for tmp in line])+'\n')
                stdout.flush()
 
        if cnt_iter%num_batches_per_epoch == 0:
            train_loss /= total_cnt
            train_t += time() - t
            
            val_loss, auroc, auprc, prevalence = eval_loss_and_prob(sess, loss, proba, data, tf_holder, batch_size)
            
            epoch_t += time() - t 
            cnt_epoch += 1
            
            reset = True

            with open(join(res_path, 'training_info.csv'), 'a') as f:
                line = [cnt_epoch, train_loss, val_loss, auroc, auprc, prevalence, train_t, epoch_t]
                writer = csv.writer(f)
                writer.writerow(line)
                stdout.write(' '.join(['%4.4g'%tmp for tmp in line])+'\n')
                stdout.flush()

            if early_stop_by == 'objective':
                update_saved_model = True if min_loss > val_loss else False
            else:
                update_saved_model = True if max_auprc < auprc else False
                
            if update_saved_model:
                cnt_wait_epoch = 0
                if early_stop_by == 'objective':
                    min_loss = val_loss
                else:
                    max_auprc = auprc
                save_path = saver.save(sess, join(res_path, 'epoch'), global_step=cnt_epoch)
            else:
                cnt_wait_epoch += 1
                if cnt_wait_epoch == max_wait_epochs:
                    saver.restore(sess, save_path)
                    stop_training = True
                    
    stdout.write('Finish training.\n')
    stdout.flush()
    
    # del train_X, train_y, train_s
    # gc.collect()
    
    del val_X, val_y, val_s
    gc.collect()

    test_X, test_y, test_s, test_dt, test_info = dataloader.load_test_data()
    data=dict(X=test_X, y=test_y, s=test_s)
    all_prob = eval_prob(sess, proba, data, tf_holder, batch_size, 'test', res_path)

    if not exists(join(res_path, 'proba')):
        mkdir(join(res_path, 'proba'))
    t = time()
    for i in range(len(test_X)):
        df = pd.DataFrame(np.hstack((test_y[i], all_prob[i])), 
                          columns=['TrueLabel', 'PredScore'])
        df['AbsDatetime'] = test_dt[i]
        df['AbsDatetime'] = pd.to_datetime(df.AbsDatetime, unit='ns')
        df.set_index('AbsDatetime', drop=True, inplace=True)
        pid, f = test_info[i]
        df.to_hdf(join(res_path, 'proba', f), pid)
        if (i+1)%10 == 0:
            stdout.write('%d patients\' scores have been written to disk. Time:%4.4g sec.\n'%((i+1), time()-t))
            stdout.flush()


    sess.close()

