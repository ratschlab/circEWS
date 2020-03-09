''' An experiment testing linearly interpolating the predictions of the MIMIC and HIRID models for fine-tuning'''

import argparse
import ipdb
import random
import os
import os.path
import pickle
import csv
import glob
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
current_palette = sns.color_palette()

import scipy
import pandas as pd
import numpy as np
import numpy.random as np_rand
import sklearn.metrics as skmetrics

import circews.functions.util.io as mlhc_io
import circews.functions.util.array as mlhc_array
import circews.functions.util.filesystem as mlhc_fs

def score_metrics(labels, scores, correct_factor=None):
    taus=[]
    tps=[]
    fps=[]
    nps=[]

    for tau in np.arange(0.00,1.01,0.01):
        der_labels=(scores>=tau).astype(np.int)
        taus.append(tau)
        tp=np.sum((labels==1.0) & (der_labels==1.0))
        npos=np.sum(labels==1.0)
        fp=np.sum((labels==0.0) & (der_labels==1.0))
        tps.append(tp)
        fps.append(fp)
        nps.append(npos)

    tps=np.array(tps)
    fps=np.array(fps)
    taus=np.array(taus)

    recalls=tps/nps
    precisions=tps/(tps+correct_factor*fps)
    precisions[np.isnan(precisions)]=1.0
        
    return (precisions, recalls, taus)

def interpolated_mimic_hirid(configs):
    static_cols_without_encode=["Age","Height","Emergency"]
    static_cols_one_hot_encode=["Surgical","APACHEPatGroup"]
    static_cols_one_hot_encode_str=["Sex"]
    str_to_int_dict={"M": 0, "F": 1, "U": 2}    
    
    random.seed(configs["random_state"])
    np_rand.seed(configs["random_state"])                
    held_out=configs["val_type"]
    dim_reduced_str=configs["data_mode"]
    task_key=configs["task_key"]
    left_hours=configs["lhours"]
    right_hours=configs["rhours"]
    val_type=configs["val_type"]
    assert(dim_reduced_str in ["reduced","non_reduced"])

    feat_order=None

    if dim_reduced_str=="reduced":
        dim_reduced_data=True
    else:
        dim_reduced_data=False    

    batch_map=mlhc_io.load_pickle(configs["mimic_pid_map_path"])["pid_to_chunk"]
    n_skipped_patients=0
    scores_dict={}
    labels_dict={}

    cal_scores_dict={}
    cal_labels_dict={}

    hirid_ml_model,hirid_col_desc,hirid_split_key=("lightgbm", "shap_top20_variables_MIMIC","held_out")
    hirid_model_dir=os.path.join(configs["predictions_dir"],"reduced",hirid_split_key,"{}_{}_{}_{}_{}".format(task_key, left_hours, right_hours, hirid_col_desc, hirid_ml_model))
    hirid_model_dir=hirid_model_dir+"_full"

    with open(os.path.join(hirid_model_dir,"best_model.pickle"),'rb') as fp:
        hirid_model=pickle.load(fp)        

    hirid_feat_order=list(hirid_model._Booster.feature_name())
    
    all_labels=[("lightgbm", "shap_top20_variables_MIMIConly_random_0","random_0"),("lightgbm", "shap_top20_variables_MIMIConly_random_1","random_1"),("lightgbm", "shap_top20_variables_MIMIConly_random_2","random_2"),
                ("lightgbm", "shap_top20_variables_MIMIConly_random_3","random_3"),("lightgbm", "shap_top20_variables_MIMIConly_random_4","random_4")]

    for mimic_ml_model, mimic_col_desc,mimic_split_key in all_labels:
        configs["split_key"]=mimic_split_key
        print("Analyzing model ({},{},{})".format(mimic_ml_model,mimic_col_desc, mimic_split_key),flush=True)
        mimic_data_split=mlhc_io.load_pickle(configs["mimic_split_path"])[mimic_split_key]        
        pred_pids=mimic_data_split[val_type]
        
        print("Number of test PIDs: {}".format(len(pred_pids)),flush=True)
        
        mimic_model_dir=os.path.join(configs["predictions_dir"],"reduced",hirid_split_key,"{}_{}_{}_{}_{}".format(task_key, left_hours, right_hours, mimic_col_desc, mimic_ml_model))
        
        feat_dir=os.path.join(configs["mimic_ml_input_dir"],"reduced",hirid_split_key,"AllLabels_0.0_8.0","X")
        labels_dir=os.path.join(configs["mimic_ml_input_dir"],"reduced",hirid_split_key,"AllLabels_0.0_8.0","y")
        impute_dir=os.path.join(configs["mimic_imputed_dir"], "reduced",hirid_split_key)
        mimic_model_dir=mimic_model_dir+"_full"

        with open(os.path.join(mimic_model_dir,"best_model.pickle"),'rb') as fp:
            mimic_model=pickle.load(fp)            

        mimic_feat_order=list(mimic_model._Booster.feature_name())
        assert(hirid_feat_order==mimic_feat_order)

        cum_pred_scores=[]
        cum_labels=[]

        cum_pred_scores_valid=[]
        cum_labels_valid=[]

        cum_pred_scores_retrain=[]
        cum_labels_retrain=[]

        df_shapelet_path=os.path.join(configs["mimic_shapelets_path"])

        n_valid_count=0

        skip_reason_key=skip_reason_ns_bef=skip_reason_ns_after=skip_reason_shapelet=0

        if configs["val_type"]=="val" or configs["full_explore_mode"]:
            ip_coeff=configs["ip_coeff"]
        else:
            val_results=glob.glob(os.path.join(configs["result_dir"],"result_val_*.tsv"))
            val_dict={}
            for rpath in sorted(val_results):
                ip_coeff_val=float(rpath.split("/")[-1].split("_")[-1][:-4])
                with open(rpath,'r') as fp:
                    csv_fp=csv.reader(fp)
                    next(csv_fp)
                    for split,auroc,auprc in csv_fp:
                        if not split==mimic_split_key:
                            continue
                        val_dict[ip_coeff_val]=float(auprc)
            ip_coeff=max(val_dict,key=val_dict.get)
            print("Best IP coeff on val set: {}".format(ip_coeff),flush=True)
        
        for pidx,pid in enumerate(pred_pids):

            if (pidx+1)%100==0 and configs["verbose"]:
                print("{}/{}, KEY: {}, NS BEF: {}, NS AFT: {}, SHAPELET: {}".format(pidx+1,len(pred_pids), skip_reason_key, skip_reason_ns_bef,skip_reason_ns_after, skip_reason_shapelet),flush=True)

            if pidx>=100 and configs["debug_mode"]:
                break
                
            batch_pat=batch_map[pid]

            try:
                pat_df=pd.read_hdf(os.path.join(feat_dir,"batch_{}.h5".format(batch_pat)), "/{}".format(pid), mode='r')
                pat_label_df=pd.read_hdf(os.path.join(labels_dir,"batch_{}.h5".format(batch_pat)), "/{}".format(pid),mode='r')
                assert(pat_df.shape[0]==pat_label_df.shape[0])
                df_feat_valid=pat_df[pat_df["SampleStatus_WorseStateFromZero0.0To8.0Hours"]=="VALID"]
                df_label_valid=pat_label_df[pat_label_df["SampleStatus_WorseStateFromZero0.0To8.0Hours"]=="VALID"]
                assert(df_feat_valid.shape[0]==df_label_valid.shape[0])
                
            except KeyError:
                skip_reason_key+=1
                continue

            if df_feat_valid.shape[0]==0:
                skip_reason_ns_bef+=1
                continue

            shapelet_df=pd.read_hdf(df_shapelet_path, '/{}'.format(pid), mode='r')
            shapelet_df["AbsDatetime"]=pd.to_datetime(shapelet_df["AbsDatetime"])
            special_cols=["AbsDatetime","PatientID"]
            shapelet_cols=list(filter(lambda col: "_dist-set" in col, sorted(shapelet_df.columns.values.tolist())))
            shapelet_df=shapelet_df[special_cols+shapelet_cols]

            if shapelet_df.shape[0]==0:
                skip_reason_shapelet+=1
                continue            

            df_merged=pd.merge(df_feat_valid,shapelet_df,on=["AbsDatetime","PatientID"])
            df_feat_valid=df_merged
            pat_label_df_orig_cols=sorted(df_label_valid.columns.values.tolist())
            df_label_valid=pd.merge(df_label_valid,shapelet_df,on=["AbsDatetime","PatientID"])
            df_label_valid=df_label_valid[pat_label_df_orig_cols]

            if df_feat_valid.shape[0]==0:
                skip_reason_ns_after+=1
                continue
            
            all_feat_cols=sorted(df_feat_valid.columns.values.tolist())
            sel_feat_cols=list(filter(lambda col: "Patient" not in col, all_feat_cols))
            X_df=df_feat_valid[sel_feat_cols]
            
            true_labels=df_label_valid["Label_WorseStateFromZero0.0To8.0Hours"]
            assert(true_labels.shape[0]==X_df.shape[0])
            X_feats=X_df[hirid_feat_order]
            X_full_collect=[X_feats]
            X_full=np.concatenate(X_full_collect,axis=1)
            
            pred_scores_mimic=mimic_model.predict_proba(X_full)[:,1]
            pred_scores_hirid=hirid_model.predict_proba(X_full)[:,1]
            
            pred_scores_ip=ip_coeff*pred_scores_hirid+(1-ip_coeff)*pred_scores_mimic

            df_out_dict={}
            abs_dt=pat_df["AbsDatetime"]
            rel_dt=pat_df["RelDatetime"]
            pred_ip_vect=mlhc_array.empty_nan(abs_dt.size)
            pred_ip_vect[pat_df["SampleStatus_WorseStateFromZero0.0To8.0Hours"]=="VALID"]=pred_scores_ip
            pred_mimic_vect=mlhc_array.empty_nan(abs_dt.size)
            pred_mimic_vect[pat_df["SampleStatus_WorseStateFromZero0.0To8.0Hours"]=="VALID"]=pred_scores_mimic
            pred_hirid_vect=mlhc_array.empty_nan(abs_dt.size)
            pred_hirid_vect[pat_df["SampleStatus_WorseStateFromZero0.0To8.0Hours"]=="VALID"]=pred_scores_hirid
            pid_vect=mlhc_array.value_empty(abs_dt.size,pid)
            y_vect=np.array(pat_label_df["Label_WorseStateFromZero0.0To8.0Hours"])
            df_out_dict["PatientID"]=pid_vect
            df_out_dict["PredScoreInterpolated"]=pred_ip_vect
            df_out_dict["PredScoreHiRiD"]=pred_hirid_vect
            df_out_dict["PredScoreMIMIC"]=pred_mimic_vect
            df_out_dict["TrueLabel"]=y_vect
            df_out_dict["AbsDatetime"]=abs_dt
            df_out_dict["RelDatetime"]=rel_dt
            df_out=pd.DataFrame(df_out_dict)
            out_dir=os.path.join(configs["result_dir"],"full_{}_set_results".format(configs["val_type"]),str(ip_coeff),mimic_split_key)
            mlhc_fs.create_dir_if_not_exist(out_dir,recursive=True)
            df_out_path=os.path.join(configs["result_dir"],"full_{}_set_results".format(configs["val_type"]),str(ip_coeff),mimic_split_key,"batch_{}.h5".format(batch_pat))

            if configs["write_output"]:
                df_out.to_hdf(df_out_path,"/p{}".format(pid),complevel=5,complib="blosc:lz4",fletcher32=True)
            
            cum_pred_scores.append(pred_scores_ip)
            cum_labels.append(true_labels)

            cum_pred_scores_valid.append(pred_scores_hirid)
            cum_labels_valid.append(true_labels)

            cum_pred_scores_retrain.append(pred_scores_mimic)
            cum_labels_retrain.append(true_labels)

            n_valid_count+=1

        scores_dict[(mimic_ml_model,mimic_col_desc,mimic_split_key,"interpolated")]=np.concatenate(cum_pred_scores)
        labels_dict[(mimic_ml_model,mimic_col_desc,mimic_split_key,"interpolated")]=np.concatenate(cum_labels)

        scores_dict[(mimic_ml_model,mimic_col_desc,mimic_split_key,"valid")]=np.concatenate(cum_pred_scores_valid)
        labels_dict[(mimic_ml_model,mimic_col_desc,mimic_split_key,"valid")]=np.concatenate(cum_labels_valid)

        scores_dict[(mimic_ml_model,mimic_col_desc,mimic_split_key,"retrain")]=np.concatenate(cum_pred_scores_retrain)
        labels_dict[(mimic_ml_model,mimic_col_desc,mimic_split_key,"retrain")]=np.concatenate(cum_labels_retrain)    

        print("Number of processed prediction set PIDs: {}/{}".format(n_valid_count,len(pred_pids)),flush=True)

    if configs["plot_type"]=="NONE":
        sys.exit(0)
        
    if configs["val_type"]=="test":
        fpath=os.path.join(configs["result_dir"],"result_{}.tsv".format(configs["val_type"]))
    else:
        fpath=os.path.join(configs["result_dir"],"result_{}_{}.tsv".format(configs["val_type"], configs["ip_coeff"]))

    color_dict={"interpolated": "C0", "valid": "C1", "retrain": "C2"}
    name_dict={"interpolated": "Interpolated", "valid": "MIMICval", "retrain": "MIMICretrain"}
        
    with open(fpath,'w') as fp:
        csv_fp=csv.writer(fp)
        csv_fp.writerow(["split","auroc","auprc","model_key"])

        for model_key in ["interpolated","valid","retrain"]:
        
            all_aurocs=[]
            all_auprcs=[]

            fpr_grid=None
            tprs=[]

            recall_grid=None
            precs=[]            

            for split in ["random_0", "random_1", "random_2", "random_3", "random_4"]:
                labels=labels_dict[("lightgbm", "shap_top20_variables_MIMIConly_{}".format(split),split,model_key)]
                scores=scores_dict[("lightgbm", "shap_top20_variables_MIMIConly_{}".format(split),split,model_key)]

                split_prevalence=np.sum(labels==1.0)/labels.size
                prevalence_bern=configs["target_prevalence_bern"]
                local_correct_factor=(1-prevalence_bern)*split_prevalence/ (prevalence_bern*(1-split_prevalence) )
                fpr_split,tpr_split,_=skmetrics.roc_curve(labels,scores)
                precs_split,recalls_split,_=score_metrics(labels,scores,correct_factor=local_correct_factor)

                if fpr_grid is None:
                    fpr_grid=fpr_split

                if recall_grid is None:
                    recall_grid=recalls_split

                tprs.append(scipy.interp(fpr_grid,fpr_split,tpr_split))
                precs.append(scipy.interp(recall_grid,recalls_split[::-1],precs_split[::-1]))

                auroc=skmetrics.roc_auc_score(labels,scores)
                auprc=skmetrics.auc(recalls_split,precs_split)

                all_aurocs.append(auroc)
                all_auprcs.append(auprc)

                csv_fp.writerow([split,str(auroc), str(auprc),model_key])

            mean_tprs=np.mean(tprs,axis=0)
            std_tprs=np.std(tprs,axis=0)
            tprs_lower=np.maximum(mean_tprs-std_tprs,0)
            tprs_upper=np.minimum(mean_tprs+std_tprs,1)

            mean_precs=np.mean(precs,axis=0)
            std_precs=np.std(precs,axis=0)
            precs_lower=np.maximum(mean_precs-std_precs,0)
            precs_upper=np.minimum(mean_precs+std_precs,1)

            if configs["plot_type"]=="roc":
                plt.plot(fpr_grid,mean_tprs,color=color_dict[model_key],label="{}, AUROC: {:.3f} ({:.3f})".format(name_dict[model_key],np.mean(all_aurocs),np.std(all_aurocs)))
                plt.fill_between(fpr_grid,tprs_lower,tprs_upper,color=color_dict[model_key],alpha=0.2)
            else:
                plt.plot(recall_grid,mean_precs,color=color_dict[model_key],label="{}: AUPRC: {:.3f} ({:.3f})".format(name_dict[model_key],np.mean(all_auprcs),np.std(all_auprcs)))
                plt.fill_between(recall_grid,precs_lower,precs_upper,color=color_dict[model_key],alpha=0.2)

        if configs["plot_type"]=="roc":

            aux_curves_orig=mlhc_io.load_pickle(configs["aux_curves_path"])

            auroc_held_out=skmetrics.auc(aux_curves_orig["bern_fpr"],aux_curves_orig["bern_tpr"])
            auroc_t1=skmetrics.auc(aux_curves_orig["bern_fpr_t1"],aux_curves_orig["bern_tpr_t1"])
            auroc_t2=skmetrics.auc(aux_curves_orig["bern_fpr_t2"],aux_curves_orig["bern_tpr_t2"])
            auroc_t3=skmetrics.auc(aux_curves_orig["bern_fpr_t3"],aux_curves_orig["bern_tpr_t3"])
            auroc_t4=skmetrics.auc(aux_curves_orig["bern_fpr_t4"],aux_curves_orig["bern_tpr_t4"])
            auroc_t5=skmetrics.auc(aux_curves_orig["bern_fpr_t5"],aux_curves_orig["bern_tpr_t5"])
            std_aurocs=np.std([auroc_t1,auroc_t2,auroc_t3,auroc_t4,auroc_t5])

            ip_tprs=[]
            ip_tprs.append(scipy.interp(aux_curves_orig["bern_fpr"], aux_curves_orig["bern_fpr_t1"], aux_curves_orig["bern_tpr_t1"]))
            ip_tprs.append(scipy.interp(aux_curves_orig["bern_fpr"], aux_curves_orig["bern_fpr_t2"], aux_curves_orig["bern_tpr_t2"]))
            ip_tprs.append(scipy.interp(aux_curves_orig["bern_fpr"], aux_curves_orig["bern_fpr_t3"], aux_curves_orig["bern_tpr_t3"]))
            ip_tprs.append(scipy.interp(aux_curves_orig["bern_fpr"], aux_curves_orig["bern_fpr_t4"], aux_curves_orig["bern_tpr_t4"]))
            ip_tprs.append(scipy.interp(aux_curves_orig["bern_fpr"], aux_curves_orig["bern_fpr_t5"], aux_curves_orig["bern_tpr_t5"]))
            std_tprs=np.std(ip_tprs,axis=0)
            tprs_upper=np.minimum(aux_curves_orig["bern_tpr"]+std_tprs,1)
            tprs_lower=np.maximum(aux_curves_orig["bern_tpr"]-std_tprs,0)
            
            plt.plot(aux_curves_orig["bern_fpr"],aux_curves_orig["bern_tpr"],color="C4",label="Original HiRID, AUROC: {:.3f} ({:.3f})".format(auroc_held_out,std_aurocs))
            plt.fill_between(aux_curves_orig["bern_fpr"],tprs_lower,tprs_upper,color="C4",alpha=0.2)
            
            plt.plot([0, 1], [0, 1], color='grey', lw=0.5, linestyle='--',rasterized=True)
            ax=plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_aspect(1.0)
            ax.grid(which="both", lw=0.5)
            plt.xlabel('1 - Specificity')
            plt.ylabel('Sensitivity')
            plt.ylim([0.0, 1.0])
            plt.xlim([0.0, 1.0])
            plt.legend(loc="lower right")
            plt.title("Interpolated score performance")
            plt.tight_layout()
            plt.savefig(os.path.join(configs["plot_dir"],"interpolated_score_roc.pdf"),bbox_inches="tight",dpi=1200,transparent=True)
            plt.savefig(os.path.join(configs["plot_dir"],"interpolated_score_roc.png"),bbox_inches="tight")
            plt.clf()

        elif configs["plot_type"]=="prc":

            aux_curves_orig=mlhc_io.load_pickle(configs["aux_curves_path_pr"])

            auprc_held_out=skmetrics.auc(aux_curves_orig["bern_recalls"],aux_curves_orig["bern_precs"])
            auprc_t1=skmetrics.auc(aux_curves_orig["bern_recalls_t1"],aux_curves_orig["bern_precs_t1"])
            auprc_t2=skmetrics.auc(aux_curves_orig["bern_recalls_t2"],aux_curves_orig["bern_precs_t2"])
            auprc_t3=skmetrics.auc(aux_curves_orig["bern_recalls_t3"],aux_curves_orig["bern_precs_t3"])
            auprc_t4=skmetrics.auc(aux_curves_orig["bern_recalls_t4"],aux_curves_orig["bern_precs_t4"])
            auprc_t5=skmetrics.auc(aux_curves_orig["bern_recalls_t5"],aux_curves_orig["bern_precs_t5"])
            std_auprcs=np.std([auprc_t1,auprc_t2,auprc_t3,auprc_t4,auprc_t5])

            ip_precs=[]
            ip_precs.append(scipy.interp(aux_curves_orig["bern_recalls"], aux_curves_orig["bern_recalls_t1"][::-1], aux_curves_orig["bern_precs_t1"][::-1]))
            ip_precs.append(scipy.interp(aux_curves_orig["bern_recalls"], aux_curves_orig["bern_recalls_t2"][::-1], aux_curves_orig["bern_precs_t2"][::-1]))
            ip_precs.append(scipy.interp(aux_curves_orig["bern_recalls"], aux_curves_orig["bern_recalls_t3"][::-1], aux_curves_orig["bern_precs_t3"][::-1]))
            ip_precs.append(scipy.interp(aux_curves_orig["bern_recalls"], aux_curves_orig["bern_recalls_t4"][::-1], aux_curves_orig["bern_precs_t4"][::-1]))
            ip_precs.append(scipy.interp(aux_curves_orig["bern_recalls"], aux_curves_orig["bern_recalls_t5"][::-1], aux_curves_orig["bern_precs_t5"][::-1]))
            std_precs=np.std(ip_precs,axis=0)
            precs_upper=np.minimum(aux_curves_orig["bern_precs"]+std_precs,1)
            precs_lower=np.maximum(aux_curves_orig["bern_precs"]-std_precs,0)
            
            plt.plot(aux_curves_orig["bern_recalls"],aux_curves_orig["bern_precs"],color="C4",label="Original HiRID, AUROC: {:.3f} ({:.3f})".format(auprc_held_out,std_auprcs))
            plt.fill_between(aux_curves_orig["bern_recalls"],precs_lower,precs_upper,color="C4",alpha=0.2)
            
            ax=plt.gca()
            ax.set_aspect(1.0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.grid(which="both", lw=0.5)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.0])
            plt.xlim([0.0, 1.0])
            plt.title('Interpolated score performance')  
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(os.path.join(configs["plot_dir"],"interpolated_score_prc.pdf"),bbox_inches="tight",dpi=1200,transparent=True)
            plt.savefig(os.path.join(configs["plot_dir"],"interpolated_score_prc.png"),bbox_inches="tight")
            plt.clf()

        else:
            print("No plot is produced...",flush=True)

def parse_cmd_args():

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--predictions_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/8_predictions/181108", help="Which predictions to analyze?")
    parser.add_argument("--mimic_pid_map_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/misc_derived/id_lists/chunks_181003.pickle", help="Path of the PID map")
    parser.add_argument("--mimic_split_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/misc_derived/split_181015.pickle", help="Path of temporal split descriptor")
    parser.add_argument("--mimic_ml_input_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/ml_input/181023")
    parser.add_argument("--mimic_imputed_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/imputed/imputed_181023", help="Imputation directory to use for the MIMIC data-set")
    parser.add_argument("--mimic_shapelets_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/9_s3m/feature_matrices_fixed/Shapelet_features_held_out_MIMIC.h5", help="Feature matrix file containing the shapelets")
    parser.add_argument("--aux_curves_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/figs_nm/Fig2/external_validation/auroc_test_WorseStateFromZero_0.0_8.0_lightgbm_mimic_external.pickle")
    parser.add_argument("--aux_curves_path_pr", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/figs_nm/time_slice/auprc_test_WorseStateFromZero_0.0_8.0_lightgbm_external_validation.pickle")

    # Output paths
    parser.add_argument("--plot_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/plots/mhueser/181108", help="Plotting base path")
    parser.add_argument("--result_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/model_interpolation", help="Validation result dir")
    parser.add_argument("--log_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/log", help="Logging dir")

    # Arguments
    parser.add_argument("--data_mode", default="reduced", help="Should reduced data be used?")
    parser.add_argument("--task_key", default="WorseStateFromZero", help="Which label should be evaluated?")
    parser.add_argument("--lhours", default=0.0, type=float, help="Left boundary of future horizon")
    parser.add_argument("--rhours", default=8.0, type=float, help="Right boundary of future horizon")
    parser.add_argument("--val_type", default="test", help="Which data set to evaluate with?")
    parser.add_argument("--verbose", default=False, action="store_true", help="Should verbose messages be printed?")
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debug mode with fewer patients")
    parser.add_argument("--plot_key", default="circEWSlite", help="Model key to analyze")
    parser.add_argument("--random_state", type=int, default=2019, help="Random state of RNG")
    parser.add_argument("--plot_type",default="NONE", help="Plot to produce")
    parser.add_argument("--write_output", default=False, action="store_true", help="Write test patient frames?")
    parser.add_argument("--target_prevalence_bern", default=0.03095, help="Target prevalence for the Bern data-set")
    parser.add_argument("--full_explore_mode", default=False, action="store_true", help="Full exploration mode which saves all the results")

    parser.add_argument("--ip_coeff", type=float, default=0.5, help="Interpolation factor between MIMIC/HIRID models")
    parser.add_argument("--run_mode", default="INTERACTIVE", help="Execute interactively or cluster?")

    configs=vars(parser.parse_args())
    return configs
            
if __name__=="__main__":
    configs=parse_cmd_args()

    if configs["run_mode"]=="CLUSTER":
        sys.stdout=open(os.path.join(configs["log_dir"],"IPCOEFF_{}_{}.stdout".format(configs["val_type"],configs["ip_coeff"])),'w')
        sys.stderr=open(os.path.join(configs["log_dir"],"IPCOEFF_{}_{}.stderr".format(configs["val_type"],configs["ip_coeff"])),'w')
    
    interpolated_mimic_hirid(configs)
