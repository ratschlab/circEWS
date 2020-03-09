import numpy as np
import pandas as pd
import h5py
from os.path import join, exists, split
from os import listdir
from sys import stdout
import gc
from time import time


class BaseDataLoader:
    bern_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital'
    def __init__(self, data_path, data_type, scaler_type, data_split, data_version, ngpus, test_mode):
        
        self.test_mode = test_mode
        if data_type not in ['top_features', 'raw_signals']:
            raise Exception('Wrong data type.')
        self.data_type = data_type
        
        # load train, val and test patient ids
        self.set_pids_ = dict()
        self.X_path = dict()
        self.y_path = dict()
        stdout.write('-------------\n')
        for set_ in ['train','val','test']:
            stdout.write('Loading patient ids for %s set. '%set_)
            stdout.flush()
            t = time()
            if self.data_type == 'top_features':
                self.X_path.update({set_:join(data_path, 'X_%s'%scaler_type, set_)})
                self.y_path.update({set_:join(data_path, 'Y', set_)})
            else:
                self.X_path.update({set_:join(data_path, scaler_type+'_signals', set_)})
                self.y_path.update({set_:join(data_path, 'labels', set_)})
            
            pids = []
            for f in listdir(self.X_path[set_]):
                with h5py.File(join(self.X_path[set_], f), 'r') as fstore:
                    pids.extend([int(key[1:]) if key[0]=='p' else int(key) for key in fstore.keys()])
            self.set_pids_.update({set_: np.sort(pids)})
            stdout.write('Finish. # patients: %d. '%len(self.set_pids_[set_]))
            stdout.write('Time: %4.4g sec. \n'%(time()-t))
            stdout.flush()

        
        self.train_pids_splits = np.array_split(self.set_pids_['train'], ngpus)
        self.permutated_train_pids_splits = dict()
        for igpu in range(ngpus):
            self.permutated_train_pids_splits.update({igpu: np.random.permutation(self.train_pids_splits[igpu])})

        stdout.write('-------------\n')
        stdout.write('Splitting training data into %d splits. \n'%ngpus)
        stdout.write('# patients in all splits (%s). \n'%(','.join([str(len(tmp)) for tmp in self.train_pids_splits])))
        stdout.write('-------------\n')
        stdout.flush()
        self.cnt_id_train = 0

    def _load_features_labels(self, set_, spec_pids=None):
        stdout.write('Loading %s data. '%set_)
        stdout.flush()
        t = time()
        X = []
        y = []
        s = []
        if set_=='test':
            dt = []
            info = []
        cnt_pids = 0
        for f in listdir(self.X_path[set_]):
            with h5py.File(join(self.X_path[set_], f), 'r') as fstore:
                pids = np.array([key for key in fstore.keys()])
                
            numeric_pids = np.array([int(pid[1:]) if pid[0]=='p' else int(pid) for pid in pids])
            if spec_pids is not None:
                is_spec_pid = np.isin(numeric_pids, spec_pids)
                pids = pids[is_spec_pid]
                numeric_pids = numeric_pids[is_spec_pid]
            
            if len(pids) == 0:
                continue

            if self.test_mode and len(pids) > 0:
                pids = pids[:20]

            for n, pid in enumerate(pids):
                
                try:
                    # load static
                    s.append( self.df_static.loc[numeric_pids[n]].values)
                    # load dynamic
                    if self.data_type == 'top_features':
                        X.append( pd.read_hdf(join(self.X_path[set_], f), pid, mode='r')[self.topk_dynamic_features].as_matrix() )
                        # with h5py.File(join(self.X_path[set_], f), 'r') as fstore:
                        #     X.append( fstore[pid][:,self.idx_topk_dynamic_features] )
                    else:
                        X.append( pd.read_hdf(join(self.X_path[set_], f), pid, mode='r').as_matrix() )
                except:
                    continue
                    

                # load label
                # df_y = pd.read_hdf(join(self.y_path[set_], f), pid, mode='r').rename(columns={'AlignDatetime':'AbsDatetime'})
                df_y = pd.read_hdf(join(self.y_path[set_], f), pid, mode='r').reset_index()
                y.append( df_y[['Label_WorseStateFromZero0.0To8.0Hours']].as_matrix() )

                # load datetime, pid, batch file info for test set
                if set_ == 'test':
                    dt.append(df_y.AbsDatetime.values if self.data_type=='top_features' else df_y.index)
                    info.append( [pid, f] )
                    
                cnt_pids += 1
                gc.collect()
                
            if self.test_mode and cnt_pids >= 20:
                break
                
        X = np.array(X)
        y = np.array(y)
        s = np.array(s)
        stdout.write('Finish. # patients: %d. '%cnt_pids)
        stdout.write('Time: %4.4g sec. \n'%(time()-t))
        stdout.flush()
        
        if set_=='test':
            dt = np.array(dt)
            info = np.array(info)
            return X, y, s, dt, info
        else:
            return X, y, s
        


    def load_train_data(self, igpu):
        return self._load_features_labels('train', spec_pids=self.train_pids_splits[igpu])

    def load_train_data_batch(self, igpu, batch_size):
        cnt_id_train_end = min(self.cnt_id_train + batch_size, len(self.train_pids_splits[igpu]))
        train_pids_batch = self.permutated_train_pids_splits[igpu][self.cnt_id_train:cnt_id_train_end]
        if cnt_id_train_end == len(self.train_pids_splits[igpu]):
            self.permutated_train_pids_splits[igpu] = np.random.permutation(self.permutated_train_pids_splits[igpu])
        self.cnt_id_train = 0 if cnt_id_train_end==len(self.train_pids_splits[igpu]) else cnt_id_train_end
        return self._load_features_labels('train', spec_pids=train_pids_batch)
    
    def load_val_data(self):
        return self._load_features_labels('val', spec_pids=self.set_pids_['val'])
    
    def load_test_data(self):
        return self._load_features_labels('test', spec_pids=self.set_pids_['test'])


class TopFeatures(BaseDataLoader):
    bern_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital'
    def __init__(self, 
                 topk=500,
                 topk_type='feature',
                 scaler_type='std',
                 data_split='temporal_5',
                 data_version='180918',
                 ngpus=1,
                 test_mode=True):
        
        scaler_type += '_scaled'
        data_path = join(self.bern_path, '7_ml_input', 'lstm_%s'%data_version, 'reduced', data_split, 'all_features')
        super(TopFeatures, self).__init__(data_path, 'top_features', scaler_type, data_split, data_version, ngpus, test_mode)

        ## Load top-k features
        all_features = np.load(join(data_path, 'feature_columns.npy'))
        topk_features = np.load(join(self.bern_path, '7_ml_input', 'lstm_%s'%data_version, 'reduced', 'top%d_%s.npz'%(topk, 'variables' if topk_type=='variable' else 'features')))
            
        # load dynamic features
        is_topk_dynamic_features = np.vectorize(lambda x: np.sum([(tmp in x if 'mode' in x else x==tmp) for tmp in topk_features['dynamic']]) > 0)
        idx_topk_dynamic_features = np.where(is_topk_dynamic_features(all_features))[0]
        self.topk_dynamic_features = all_features[idx_topk_dynamic_features]
        # load static features
        df_static = pd.read_hdf(join(data_path, 'X_%s'%scaler_type, 'static.h5'), mode='r')
        df_static = df_static[df_static.index.isin(np.concatenate([val for _, val in self.set_pids_.items()]))]
        if len(topk_features['static']) > 0:
            static_features = [col for col in df_static.columns if np.sum([f in col for f in topk_features['static']]) > 0]
            self.df_static = df_static[static_features]
            



class RawSignals(BaseDataLoader):
    bern_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital'
    def __init__(self, 
                 scaler_type='std',
                 data_split='temporal_5',
                 data_version='180918',
                 ngpus=1,
                 test_mode=True):
        scaler_type += '_scaled'
        data_path = join(self.bern_path, '7_ml_input', 'lstm', '%s_%s'%(data_version, data_split))
        super(RawSignals, self).__init__(data_path, 'raw_signals', scaler_type, data_split, data_version, ngpus,test_mode)

        # load static features
        df_static = pd.read_hdf(join(data_path, 'X_%s'%scaler_type, 'static.h5'), mode='r')
        self.df_static = df_static[df_static.index.isin(np.concatenate([val for _, val in self.set_pids_.items()]))]
                

if __name__=='__main__':
    tmp = RawSignals()
    tmp = TopFeatures()
