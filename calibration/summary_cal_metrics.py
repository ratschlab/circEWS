''' Summarize calibration summary metrics from the data files'''

import argparse
import glob
import os
import os.path

import circews.functions.util.io as mlhc_io

def print_cal_summary_metrics(configs):

    CATEGORIES=["ApacheScoreGroup_range"]
    MODELS=["shap_top500_features","shap_top20_variables_MIMIC_BERN"]

    for model in MODELS:
        for cate in CATEGORIES:
            fpaths=sorted(glob.glob(os.path.join(configs["data_in_dir"],"*_{}_*.pickle".format(cate))))
            fpaths=list(filter(lambda fpath: model in fpath, fpaths))
            fpaths=list(filter(lambda fpath: "_Surgical_" not in fpath, fpaths))

            for fpath in fpaths:
                last_part=fpath.split("/")[-1].strip()
                metric_dict=mlhc_io.load_pickle(fpath)
                print("File: {}".format(last_part))
                print("Raw Gini index Mean: {:.3f}, Std: {:.3f}".format(metric_dict["raw_gini_mean"], metric_dict["raw_gini_std"]))                
                print("Cal. Gini index Mean: {:.3f}, Std: {:.3f}".format(metric_dict["iso_gini_mean"], metric_dict["iso_gini_std"]))
        

def parse_cmd_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_in_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/model_calibration",
                        help="Data input directory")
    configs=vars(parser.parse_args())    
    return configs


if __name__=="__main__":
    configs=parse_cmd_args()
    print_cal_summary_metrics(configs)
    
