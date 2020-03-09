import tensorflow as tf
import numpy as np
from sys import stdout
import sklearn.metrics as sk_metrics
from os.path import join

def sequence2matrix(seq_dynamic, seq_output, batch_size):
    if batch_size > 1:
        seq_len = np.ones((batch_size,), dtype=int)*2
        for i in range(len(seq_dynamic)):
            seq_len[i] = len(seq_dynamic[i])

        mat_dynamic = np.zeros((batch_size, seq_len.max(), seq_dynamic[0].shape[1]))
        mat_output = np.zeros((batch_size, seq_len.max(), seq_output[0].shape[1]))
        mat_mask = np.zeros((batch_size, seq_len.max()), dtype=bool)
        for i in range(len(seq_dynamic)):
            mat_dynamic[i,:seq_len[i],:] = seq_dynamic[i]
            mat_output[i,:seq_len[i],:] = seq_output[i]
            mat_mask[i,:seq_len[i]] = True
            index_nan = np.where(seq_output[i]==2)[0]
            if len(index_nan) > 0:
                mat_mask[i,index_nan] = False
    else:
        mat_dynamic = np.stack(seq_dynamic, axis=0)
        mat_output = np.stack(seq_output, axis=0)
        mat_mask = np.ones((1, len(seq_dynamic[0])), dtype=bool)
        mat_mask[mat_output[0].reshape((1,-1))==2] = False
        seq_len = [len(seq_dynamic[0])]
        
    return mat_dynamic, mat_output, mat_mask, seq_len        


def tf_proba(tf_var, tf_holder, activ_func):
    hidden_dynamic, _ = tf.nn.dynamic_rnn(tf_var['lstm'],
                                          tf.nn.dropout(tf_holder['D_'], keep_prob=tf_holder['keep_prob_']),
                                          sequence_length=tf_holder['seq_len_'],
                                          dtype=tf.float32)
    if activ_func == 'tanh':
        tf_activ_func = tf.tanh
    elif activ_func == 'relu':
        tf_activ_func = tf.nn.relu
    elif activ_func == 'sigmoid':
        tf_activ_func = tf.nn.sigmoid
        

    hidden_static = tf_activ_func(tf.matmul(tf.nn.dropout(tf_holder['S_'], keep_prob=tf_holder['keep_prob_']),
                                            tf_var['Ws0']) + tf_var['bs0'] )

    hidden_concat = tf.concat([hidden_dynamic, tf.tile(tf.expand_dims(hidden_static, 1), [1, tf.shape(hidden_dynamic)[1], 1])], axis=2)

    
    hidden_joint = tf_activ_func(tf.einsum('nth,ho->nto', tf.nn.dropout(hidden_concat, keep_prob=tf_holder['keep_prob_']), tf_var['Wj']) + tf_var['bj'])
    output = tf.einsum('nth,ho->nto', tf.nn.dropout(hidden_joint, keep_prob=tf_holder['keep_prob_']), tf_var['Wo']) + tf_var['bo']
    
    proba = tf.nn.sigmoid(output)

    return output, proba


def eval_prob(sess, proba, data, tf_holder, batch_size, set_, res_path):
    cnt = 0
    all_prob = []
    data_size = len(data['X'])
    while cnt < data_size:
        batch_dynamic = data['X'][cnt:min(cnt+batch_size, data_size)]
        batch_static = data['s'][cnt:min(cnt+batch_size, data_size)]
        batch_output = data['y'][cnt:min(cnt+batch_size, data_size)]

        cnt = min(cnt+batch_size, data_size)
        mat_dynamic, _, _, seq_len = sequence2matrix(batch_dynamic, batch_output, batch_size)
        if len(batch_static) < batch_size:
            batch_static = np.vstack((batch_static, 
                                      np.zeros((batch_size-len(batch_static), batch_static.shape[1]))))

        mat_prob = sess.run(proba, feed_dict={tf_holder['D_']: mat_dynamic, 
                                              tf_holder['S_']: batch_static, 
                                              tf_holder['seq_len_']: seq_len,
                                              tf_holder['keep_prob_']: 1.})
        mat_prob = mat_prob[:len(batch_dynamic)]
        seq_prob = [mat_prob[i,:seq_len[i],:] for i in range(len(mat_prob))]
        all_prob.extend(seq_prob)

    y_true = np.vstack(tuple(data['y']))
    y_score =  np.vstack(tuple(all_prob))   
    y_score = y_score[y_true < 2]
    y_true = y_true[y_true < 2]
    auroc = sk_metrics.roc_auc_score(y_true, y_score)
    auprc = sk_metrics.average_precision_score(y_true, y_score)
    prevalence = np.sum(y_true) / len(y_true)
    stdout.write('------ %s -----\n'%set_.upper())
    stdout.write(' '.join(['auroc', 'auprc', 'prevalence'])+'\n')
    stdout.write( ' '.join(['%4.4g'%tmp for tmp in [auroc, auprc, prevalence]])+'\n')
    stdout.flush()
    np.savez(join(res_path, '%s_scores.npz'%set_), auroc=auroc, auprc=auprc, prevalence=prevalence)
    return all_prob


def eval_loss_and_prob(sess, loss, proba, data, tf_holder, batch_size):
    cnt = 0
    total_loss = 0
    total_cnt = 0
    all_prob = []
    data_size = len(data['X'])
    while cnt < data_size:
        batch_dynamic = data['X'][cnt:min(cnt+batch_size, data_size)]
        batch_static = data['s'][cnt:min(cnt+batch_size, data_size)]
        batch_output = data['y'][cnt:min(cnt+batch_size, data_size)]

        cnt = min(cnt+batch_size, data_size)
        mat_dynamic, mat_output, mat_mask, seq_len = sequence2matrix(batch_dynamic, batch_output, batch_size)
        if len(batch_static) < batch_size:
            batch_static = np.vstack((batch_static, 
                                      np.zeros((batch_size-len(batch_static), batch_static.shape[1]))))

        batch_loss, mat_prob = sess.run([loss, proba], feed_dict={tf_holder['D_']: mat_dynamic, 
                                                                  tf_holder['y_']: mat_output, 
                                                                  tf_holder['S_']: batch_static, 
                                                                  tf_holder['mask']: mat_mask,
                                                                  tf_holder['seq_len_']: seq_len,
                                                                  tf_holder['keep_prob_']: 1.})
        mat_prob = mat_prob[:len(batch_dynamic)]
        seq_prob = [mat_prob[i,:seq_len[i],:] for i in range(len(mat_prob))]
        all_prob.extend(seq_prob)


        if not np.isnan(batch_loss):
            total_loss += batch_loss * len(batch_dynamic)
            total_cnt += len(batch_dynamic)
    total_loss /= total_cnt
    
    y_true = np.vstack(tuple(data['y']))
    y_score =  np.vstack(tuple(all_prob)) 
    y_score = y_score[y_true < 2]
    y_true = y_true[y_true < 2]
    auroc = sk_metrics.roc_auc_score(y_true, y_score)
    auprc = sk_metrics.average_precision_score(y_true, y_score)
    prevalence = np.sum(y_true) / len(y_true)
    
    return total_loss, auroc, auprc, prevalence
