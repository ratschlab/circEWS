import pandas as pd
import numpy as np
import h5py


def get_resolutions(data_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/7_ml_input/180918/reduced/temporal_5/AllLabels_0.0_8.0/X/batch_8.h5'):
    with h5py.File(data_path, 'r') as f:
        pids = [key for key in f.keys()]
    df = pd.read_hdf(data_path, pids[0])

    high_freq_f = np.unique([col.split('_')[2] for col in df.columns if 'high_' in col])

    med_freq_f = np.unique([col.split('_')[2] for col in df.columns if 'med_' in col])

    low_freq_f = np.unique([col.split('_')[2] for col in df.columns if 'low_' in col])

    return high_freq_f,med_freq_f,low_freq_f

if __name__ == '__main__':
    high_freq_f,med_freq_f,low_freq_f = get_resolutions()

    print(high_freq_f)
    #array(['pm39', 'pm40', 'pm41', 'pm42', 'pm45', 'pm46', 'pm53', 'pm78', 'pm80', 'pm86', 'vm1', 'vm10', 'vm11', 'vm13', 'vm14', 'vm15', 'vm16', 'vm17', 'vm18', 'vm20', 'vm21', 'vm22', 'vm29', 'vm3', 'vm31', 'vm33', 'vm34', 'vm4', 'vm5', 'vm58', 'vm59', 'vm6', 'vm60', 'vm61', 'vm62', 'vm63', 'vm64', 'vm65', 'vm7', 'vm8', 'vm9'], dtype='<U4')


    print(med_freq_f)
    #array(['pm100', 'pm112', 'pm113', 'pm117', 'pm119', 'pm120', 'pm121', 'pm122', 'pm35', 'pm36', 'pm37', 'pm38', 'pm44', 'pm47', 'pm56', 'pm71', 'pm77', 'pm79', 'pm81', 'pm83', 'pm85', 'pm87', 'pm88', 'pm89', 'pm90', 'pm95', 'pm97', 'pm99', 'vm12', 'vm132', 'vm133', 'vm134', 'vm135', 'vm136', 'vm137', 'vm138', 'vm139', 'vm140', 'vm141', 'vm142', 'vm148', 'vm149', 'vm150', 'vm151', 'vm174', 'vm19', 'vm2', 'vm23', 'vm24', 'vm25', 'vm26', 'vm27', 'vm28', 'vm30', 'vm32', 'vm66', 'vm84'], dtype='<U5')

    print(low_feq_f)
    #array(['pm101', 'pm102', 'pm103', 'pm104', 'pm105', 'pm106', 'pm107', 'pm108', 'pm109', 'pm110', 'pm111', 'pm114', 'pm115', 'pm116', 'pm118', 'pm123', 'pm124', 'pm125', 'pm126', 'pm127', 'pm128', 'pm129', 'pm130', 'pm43', 'pm48', 'pm49', 'pm50', 'pm51', 'pm52', 'pm54', 'pm55', 'pm67', 'pm68', 'pm69', 'pm70', 'pm73', 'pm74', 'pm75', 'pm76', 'pm82', 'pm91', 'pm92', 'pm93', 'pm94', 'pm96', 'pm98', 'vm131', 'vm143', 'vm144', 'vm145', 'vm146', 'vm147', 'vm152', 'vm153', 'vm154', 'vm155', 'vm156', 'vm157', 'vm158', 'vm159', 'vm160', 'vm161', 'vm162', 'vm163', 'vm164', 'vm165', 'vm166', 'vm167', 'vm168', 'vm169', 'vm170', 'vm171', 'vm172', 'vm173', 'vm175', 'vm176', 'vm177', 'vm178', 'vm179', 'vm180', 'vm181', 'vm182', 'vm183', 'vm184', 'vm185', 'vm186', 'vm187', 'vm188', 'vm189', 'vm190', 'vm191', 'vm192', 'vm193', 'vm194', 'vm195', 'vm196', 'vm197', 'vm198'], dtype='<U5')

