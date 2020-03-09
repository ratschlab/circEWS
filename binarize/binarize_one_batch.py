''' Binarize one batch of imputed matrices'''

import argparse
import os
import os.path
import sys

import pandas as pd
import numpy as np

def binarize_one_batch(configs):

    if configs["data_mode"]=="reduced":
        inputf=os.path.join(configs["bern_imputed_reduced_dir"],configs["split_key"],"batch_{}.h5".format(configs["batch_idx"]))
    else:
        inputf=os.path.join(configs["bern_imputed_dir"],configs["split_key"],"batch_{}.h5".format(configs["batch_idx"]))

    if not os.path.exists(inputf):
        return

    df=pd.read_hdf(inputf,mode='r')
    unique_pids=list(df.PatientID.unique())
    binarized_dir=configs["binarized_dir"]
    copy_cols=["PatientID","AbsDatetime","RelDatetime"]
    status_cols=list(filter(lambda col: "IMPUTED_STATUS" in col, df.columns.values.tolist()))
    raw_cols=list(filter(lambda col: "IMPUTED_STATUS" not in col and ("pm" in col or "vm" in col), df.columns.values.tolist()))
    assert(len(copy_cols)+len(status_cols)+len(raw_cols)==len(df.columns))
    first_write=True
    outf=os.path.join(binarized_dir,"reduced",configs["split_key"],"batch_{}.h5".format(configs["batch_idx"]))

    if os.path.exists(outf):
        os.remove(outf)

    for pix,pid in enumerate(unique_pids):

        if (pix+1)%10==0:
            print("Processing PID: {}/{}".format(pix+1,len(unique_pids)))        

        df_pid=df[df["PatientID"]==pid]
        out_dict={}

        for col in copy_cols:
            out_dict[col]=df_pid[col].copy()

        for col in raw_cols:
            in_col=np.array(df_pid[col])

            if "pm" in col: 
                in_col[in_col!=0]=1.0
            else: 
                in_col[np.isfinite(in_col)]=1.0
                
            in_col[np.isnan(in_col)]=0.0
            out_dict[col]=in_col

        for col in status_cols:
            out_dict[col]=df_pid[col].copy()            
        
        df_out_pid=pd.DataFrame(out_dict)

        if first_write:
            if not configs["debug_mode"]:
                df_out_pid.to_hdf(outf,'data',mode='w',format="table", append=False, data_columns=["PatientID"], complevel=5,complib="blosc:lz4", fletcher32=True)
            first_write=False
        else:
            if not configs["debug_mode"]:
                df_out_pid.to_hdf(outf,'data',mode='a',format="table", append=True, data_columns=["PatientID"], complevel=5,complib="blosc:lz4", fletcher32=True)

def parse_cmd_args():
    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--bern_imputed_reduced_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5_imputed/imputed_v6b_downsample_upsample_no_impute/reduced",
                        help="Input data that should be binarized, reduced version")
    parser.add_argument("--bern_imputed_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5_imputed/imputed_v6b_downsample_upsample_no_impute",
                        help="Input data that should be binarized")

    # Output paths
    parser.add_argument("--log_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/log", help="Location of the log directory")
    parser.add_argument("--binarized_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5b_binarized/binarized_v6b_downsample_upsample_rev2", 
                        help="Binarized data-set with 1 for observed values, 0 otherwise")

    # Arguments
    parser.add_argument("--run_mode", default="INTERACTIVE", help="Should job be run in batch or interactive mode")
    parser.add_argument("--data_mode", default="reduced", help="Should dim-reduced data be used?")
    parser.add_argument("--batch_idx", type=int, default=10, help="On which batch should imputation be run?")
    parser.add_argument("--split_key", default="temporal_5", help="On which split should imputation be run?")
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debug mode, not writing to file-system")

    configs=vars(parser.parse_args())
    return configs

if __name__=="__main__":
    configs=parse_cmd_args()
    split_key=configs["split_key"]
    dim_reduced_str=configs["data_mode"]
    batch_idx=configs["batch_idx"]    

    if configs["run_mode"]=="CLUSTER":
        sys.stdout=open(os.path.join(configs["log_dir"],"BINARIZE_{}_{}_{}.stdout".format(split_key,dim_reduced_str, batch_idx)),'w')
        sys.stderr=open(os.path.join(configs["log_dir"],"BINARIZE_{}_{}_{}.stderr".format(split_key,dim_reduced_str, batch_idx)),'w')

    binarize_one_batch(configs)


    
