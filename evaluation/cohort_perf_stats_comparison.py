''' Statistical comparison of performance in sub-cohort vs. the rest'''

import ipdb
import csv
import glob
import os
import os.path
import argparse

import scipy.stats as sp_stats
import mne.stats as mne_stats
import numpy as np
import statsmodels.regression.linear_model as statslm
import statsmodels.tools.tools as statstools

def cohort_1d_regressions(configs):

    for fpath in sorted(glob.glob(os.path.join(configs["metagroup_path"],"*.csv"))):
        groupname=fpath.split("/")[-1][:-4]
        print("Group: {}".format(groupname))
        with open(fpath,'r') as fp:
            csv_fp=csv.reader(fp,delimiter=',')
            next(csv_fp)
            x=[]
            y=[]
            for line in csv_fp:
                if line[0].strip()=="held_out" or line[1]=="" or line[2]=="":
                    continue
                x.extend([1,0])
                y.extend([float(line[1]),float(line[2])])

        lm=statslm.OLS(y,statstools.add_constant(np.array(x).reshape(len(x),1)),missing='raise', hasconst=True)
        results=lm.fit()
        print("Group result for group {} intercept/group effect: {}, t-test p-val: {}".format(groupname,results.params,results.pvalues))


def cohort_paired_t_tests(configs):
    all_pvals=[]
    groupnames=[]
    basepath=os.path.join(configs["base_path"],"for_{}_{}_stats_importance".format(configs["grouping"],
                                                                                   configs["comparison_metric"]))

    for fpath in sorted(glob.glob(os.path.join(basepath,"*.csv"))):
        groupname=fpath.split("/")[-1][:-4]
        a=[]
        b=[]
        with open(fpath,'r') as fp:
            csv_fp=csv.reader(fp,delimiter=',')
            next(csv_fp)
            for line in csv_fp:
                if line[1]=="" or line[2]=="":
                    continue
                a.append(float(line[1]))
                b.append(float(line[2]))

        stats,pval=sp_stats.ttest_rel(a,b)
        if np.isnan(pval):
            continue
        groupnames.append(groupname)
        all_pvals.append(pval)

    rej,corr=mne_stats.fdr_correction(all_pvals,method="indep")
    
    for idx,gname in enumerate(groupnames):
        print("Group; {}, Corrected pval: {}".format(gname,corr[idx]))

def cohort_one_regression(configs):

    group_idxs=[]
    
    for idx,fpath in enumerate(sorted(glob.glob(os.path.join(configs["metagroup_path"],"*.csv")))):
        groupname=fpath.split("/")[-1][:-4]
        group_idxs.append((idx,groupname))

    x=[]
    y=[]
        
    for fpath in sorted(glob.glob(os.path.join(configs["metagroup_path"],"*.csv"))):
        groupname=fpath.split("/")[-1][:-4]
        print("Group: {}".format(groupname))
        with open(fpath,'r') as fp:
            csv_fp=csv.reader(fp,delimiter=',')
            next(csv_fp)
            for line in csv_fp:
                if line[1]=="":
                    continue
                x.append([(1 if group_idxs[idx][1]==groupname else 0) for idx in range(len(group_idxs))])
                y.extend([float(line[1])])

    y.append(0.9)                
    y=np.array(y)
    x.append(np.zeros(len(group_idxs)))
    x=np.array(x)
    print("Size of design matrix: {}".format(x.shape))
    lm=statslm.OLS(y,statstools.add_constant(x),missing='raise', hasconst=None)
    
    results=lm.fit()

    print("Intercept, Coefficient: {}, p-value: {}".format(results.params[0], results.pvalues[0]))
    
    for idx in range(len(group_idxs)):
        pval=results.pvalues[idx+1]
        print("Group: {}, Coefficient: {}, p-value: {}".format(group_idxs[idx][1], results.params[idx+1], results.pvalues[idx+1]))
    
def parse_cmd_args():
    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--base_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/figs_nm/Fig4/cohort_performance")

    # Arguments
    parser.add_argument("--comparison_metric", default="rec", help="Which metric should be analyzed?")
    parser.add_argument("--grouping", default="meta_group", help="Which meta-group should be analyzed?")
    
    args=parser.parse_args()
    configs=vars(args)
    return configs
    
if __name__=="__main__":
    configs=parse_cmd_args()
    cohort_paired_t_tests(configs)
