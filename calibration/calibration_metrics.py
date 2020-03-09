''' Evaluation off calibration metrics'''

import argparse
import os
import os.path
import ipdb
import random
import pickle
import csv

import numpy as np
import pandas as pd
import numpy.random as np_rand
import sklearn.calibration as skcal
import sklearn.metrics as skmetrics
import sklearn.linear_model as sklm
import sklearn.isotonic as skiso
import lightgbm
import scipy

import matplotlib as mpl
mpl.use("PDF")
import matplotlib.pyplot as plt

import circews.functions.util.io as mlhc_io
import circews.functions.util.array as mlhc_array
import circews.functions.features as bern_features

def rescaled_calibration_curve(y_true,y_prob, correct_factor=None, n_bins=20):
    ''' Rescaling for prevalence version of the calibration curve'''
    bin_locs=np.arange(y_prob.min(),y_prob.max(),0.05)
    act_risks=[]
    act_locs=[]

    for thr in range(1,len(bin_locs)):
        risk_lab=y_true[(y_prob>=bin_locs[thr-1]) & (y_prob<bin_locs[thr])]
        if risk_lab.size==0:
            continue
        tps=np.sum(risk_lab==1.0)
        fps=correct_factor*np.sum(risk_lab==0.0)
        act_risk=tps/(tps+fps)
        act_risks.append(act_risk)
        act_locs.append((bin_locs[thr-1]+bin_locs[thr])/2.)

    return (np.array(act_risks),np.array(act_locs))

def alarm_folder(model_desc,mimic_split_key):
    if "top500" in model_desc:
        return "shap_top500_features"
    elif "MIMIC_BERN" in model_desc:
        return "shap_top20_variables_MIMIC_BERN"
    elif "MIMIConly" in model_desc:
        return "shap_top20_variables_MIMIConly"
    else:
        return "shap_top20_variables_MIMIC"+"_no_subsample_val_on_{}".format(mimic_split_key)

def calibration_metrics(configs):

    static_cols_without_encode=["Age","Height","Emergency"]
    static_cols_one_hot_encode=["Surgical","APACHEPatGroup"]
    static_cols_one_hot_encode_str=["Sex"]
    str_to_int_dict={"M": 0, "F": 1, "U": 2}    
    
    random.seed(configs["random_state"])
    np_rand.seed(configs["random_state"])                
    held_out=configs["val_type"]
    cal_set=configs["calibration_set"]
    dim_reduced_str=configs["data_mode"]
    task_key=configs["task_key"]
    left_hours=configs["lhours"]
    right_hours=configs["rhours"]
    assert(dim_reduced_str in ["reduced","non_reduced"])
    threshold_dict={}

    with open(configs["threshold_map"],'r') as fp:
        csv_fp=csv.reader(fp)
        next(csv_fp)
        for _,split_key,recall,threshold,model in csv_fp:
            if not float(recall)==configs["desired_recall"]:
                continue
            threshold_dict[("shap_"+model,split_key)]=float(threshold)

    feat_order=None

    if dim_reduced_str=="reduced":
        dim_reduced_data=True
    else:
        dim_reduced_data=False    

    bern_batch_map=mlhc_io.load_pickle(configs["bern_pid_map_path"])["pid_to_chunk"]
    mimic_batch_map=mlhc_io.load_pickle(configs["mimic_pid_map_path"])["pid_to_chunk"]
    
    n_skipped_patients=0
    scores_dict={}
    labels_dict={}

    cal_scores_dict={}
    cal_labels_dict={}

    all_labels=[("lightgbm", "shap_top20_variables_MIMIC_BERN","held_out",None),("lightgbm", "shap_top20_variables_MIMIC_BERN","temporal_1",None),
                ("lightgbm", "shap_top20_variables_MIMIC_BERN","temporal_2",None),("lightgbm", "shap_top20_variables_MIMIC_BERN","temporal_3",None),
                ("lightgbm", "shap_top20_variables_MIMIC_BERN","temporal_4",None),("lightgbm", "shap_top20_variables_MIMIC_BERN","temporal_5",None),
                ("lightgbm","shap_top500_features","held_out",None),("lightgbm","shap_top500_features","temporal_1",None),
                ("lightgbm","shap_top500_features","temporal_2",None),("lightgbm","shap_top500_features","temporal_3",None),
                ("lightgbm","shap_top500_features","temporal_4",None),("lightgbm","shap_top500_features","temporal_5",None)]

    for ml_model, col_desc,split_key,mimic_split_key in all_labels:

        if configs["only_alarm_points"]:
            if "MIMIConly" in col_desc:
                critical_score=threshold_dict[(col_desc.strip(), mimic_split_key)]                
            else:
                critical_score=threshold_dict[(col_desc.strip(), split_key)]
        
        configs["split_key"]=split_key
        print("Analyzing model ({},{},{})".format(ml_model,col_desc, split_key))
        cum_pred_scores=[]
        cum_labels=[]
        data_split=mlhc_io.load_pickle(configs["temporal_split_path"])[split_key]

        if "MIMIC" in col_desc and "BERN" not in col_desc:
            mimic_data_split=mlhc_io.load_pickle(configs["mimic_split_path"])[mimic_split_key]

        if "MIMIC" in col_desc and "BERN" not in col_desc:
            pred_pids=mimic_data_split[held_out]
        else:
            pred_pids=data_split[held_out]

            if not configs["cohort_category"]=="global":
                pid_maps={}
                all_mapped_pids=[]
                
                with open(configs["pid_category_map"],'r') as fp:
                    csv_fp=csv.reader(fp,delimiter='\t')
                    header_map={}
                    header_line=next(csv_fp)
                    for idx,field in enumerate(header_line):
                        if field.strip() not in ["Sex","AgeGroup_range","Emergency", "ApacheGroupName","Surgical", "ApacheScoreGroup_range"]:
                            continue
                        pid_maps[field.strip()]={}
                        header_map[field.strip()]=idx
                    for line in csv_fp:
                        pid=int(line[0])
                        all_mapped_pids.append(pid)
                        for k in header_map.keys():
                            pid_maps[k][pid]=line[header_map[k]].strip()

                pred_pids=list(filter(lambda pid: pid in all_mapped_pids and pid_maps[configs["cohort_category"].strip()][pid]==configs["cohort_value"].strip(), pred_pids))

                if not configs["cohort_category_fine"]=="global":
                    pred_pids=list(filter(lambda pid: pid in all_mapped_pids and pid_maps[configs["cohort_category_fine"].strip()][pid]==configs["cohort_value_fine"].strip(), pred_pids))
                
            num_filtered_pids=len(pred_pids)
            print("Number of filtered test PIDs: {}".format(num_filtered_pids))

        if "MIMIConly" in col_desc:
            calibration_pids=mimic_data_split[cal_set]
        else:
            calibration_pids=data_split[cal_set]
        
        print("Number of test PIDs: {}".format(len(pred_pids)))
        print("Number of calibration PIDs: {}".format(len(calibration_pids)))

        output_dir=os.path.join(configs["predictions_dir"],"reduced",split_key,"{}_{}_{}_{}_{}".format(task_key, left_hours, right_hours, col_desc, ml_model))

        if "MIMIConly" in col_desc:
            feat_dir=os.path.join(configs["mimic_ml_input_dir"],"reduced",split_key,"AllLabels_0.0_8.0","X")
            labels_dir=os.path.join(configs["mimic_ml_input_dir"],"reduced",split_key,"AllLabels_0.0_8.0","y")
            impute_dir=os.path.join(configs["mimic_imputed_dir"], "reduced",split_key)
            full_static_df=pd.read_hdf(os.path.join(impute_dir,"static.h5"),mode='r')            
        else:
            feat_dir=os.path.join(configs["bern_ml_input_dir"],"reduced",split_key,"AllLabels_0.0_8.0","X")
            labels_dir=os.path.join(configs["bern_ml_input_dir"],"reduced",split_key,"AllLabels_0.0_8.0","y")
            impute_dir=os.path.join(configs["bern_imputed_dir"], "reduced",split_key)
            full_static_df=pd.read_hdf(os.path.join(impute_dir,"static.h5"),mode='r')

        output_dir=output_dir+"_full"

        with open(os.path.join(output_dir,"best_model.pickle"),'rb') as fp:
            orig_model=pickle.load(fp)

        feat_order=list(filter(lambda col: "static_" not in col, orig_model._Booster.feature_name()))

        for pidx,pid in enumerate(pred_pids):

            if (pidx+1)%500==0 and configs["verbose"]:
                print("{}/{}".format(pidx+1,len(pred_pids)))

            if pidx>=100 and configs["debug_mode"]:
                break

            if "MIMIC" in col_desc and "BERN" not in col_desc:
                batch_pat=mimic_batch_map[pid]
            else:
                batch_pat=bern_batch_map[pid]

            try:
                df_pred=pd.read_hdf(os.path.join(output_dir,"batch_{}.h5".format(batch_pat)), "/p{}".format(pid), mode='r')
                df_pred=df_pred[pd.notna(df_pred["TrueLabel"]) & pd.notna(df_pred["PredScore"])]

            except KeyError:
                n_skipped_patients+=1
                continue

            if configs["only_alarm_points"]:

                try:
                    alarm_path=alarm_folder(col_desc,mimic_split_key)
                    split_path=mimic_split_key if "MIMIConly" in col_desc else split_key
                    df_alarms=pd.read_hdf(os.path.join(configs["alarms_path"],alarm_path,split_path,"batch_{}.h5".format(batch_pat)),"/p{}".format(pid),mode='r')
                except KeyError:
                    continue
                    
                df_im=pd.merge(df_pred,df_alarms,on=["RelDatetime"])
                df_pred=df_im[df_im["Status"]=="Alarm"]
                df_pred.rename(columns={"PredScore_x": "PredScore"},inplace=True)

            pred_scores=np.array(df_pred["PredScore"])
            true_labels=np.array(df_pred["TrueLabel"])
            cum_pred_scores.append(pred_scores)
            cum_labels.append(true_labels)

        scores_dict[(ml_model,col_desc,split_key,mimic_split_key)]=np.concatenate(cum_pred_scores)
        labels_dict[(ml_model,col_desc,split_key,mimic_split_key)]=np.concatenate(cum_labels)

        cum_cal_scores=[]
        cum_cal_labels=[]

        if "MIMIConly" in col_desc:
            df_shapelet_path=os.path.join(configs["shapelets_path"],"Shapelet_features_{}_MIMIC.h5".format(split_key))
        else:
            df_shapelet_path=os.path.join(configs["shapelets_path"],"Shapelet_features_{}.h5".format(split_key))

        n_valid_count=0

        for pidx,pid in enumerate(calibration_pids):

            if (pidx+1)%500==0 and configs["verbose"]:
                print("{}/{}".format(pidx+1,len(pred_pids)))

            if pidx>=100 and configs["debug_mode"]:
                break

            if "MIMIConly" in col_desc:
                batch_pat=mimic_batch_map[pid]
            else:
                batch_pat=bern_batch_map[pid]

            try:
                pat_df=pd.read_hdf(os.path.join(feat_dir,"batch_{}.h5".format(batch_pat)), "/{}".format(pid), mode='r')
                pat_label_df=pd.read_hdf(os.path.join(labels_dir,"batch_{}.h5".format(batch_pat)), "/{}".format(pid),mode='r')
                assert(pat_df.shape[0]==pat_label_df.shape[0])
                df_feat_valid=pat_df[pat_df["SampleStatus_WorseStateFromZero0.0To8.0Hours"]=="VALID"]
                df_label_valid=pat_label_df[pat_label_df["SampleStatus_WorseStateFromZero0.0To8.0Hours"]=="VALID"]
                assert(df_feat_valid.shape[0]==df_label_valid.shape[0])
                
            except KeyError:
                continue

            if df_feat_valid.shape[0]==0:
                continue

            shapelet_df=pd.read_hdf(df_shapelet_path, '/{}'.format(pid), mode='r')
            shapelet_df["AbsDatetime"]=pd.to_datetime(shapelet_df["AbsDatetime"])
            special_cols=["AbsDatetime","PatientID"]
            shapelet_cols=list(filter(lambda col: "_dist-set" in col, sorted(shapelet_df.columns.values.tolist())))
            shapelet_df=shapelet_df[special_cols+shapelet_cols]

            if shapelet_df.shape[0]==0:
                continue            

            df_merged=pd.merge(df_feat_valid,shapelet_df,on=["AbsDatetime","PatientID"])
            df_feat_valid=df_merged
            pat_label_df_orig_cols=sorted(df_label_valid.columns.values.tolist())
            df_label_valid=pd.merge(df_label_valid,shapelet_df,on=["AbsDatetime","PatientID"])
            df_label_valid=df_label_valid[pat_label_df_orig_cols]

            if df_feat_valid.shape[0]==0:
                continue
            
            all_feat_cols=sorted(df_feat_valid.columns.values.tolist())
            sel_feat_cols=list(filter(lambda col: "Patient" not in col, all_feat_cols))
            X_df=df_feat_valid[sel_feat_cols]
            
            static_df=full_static_df[full_static_df["PatientID"]==pid]

            if static_df.shape[0]<1:
                n_skipped_patients+=1
                skip_no_staticdf+=1
                continue
            
            final_cols=bern_features.yield_final_cols(col_desc,sel_feat_cols, configs)
            final_static_cols=list(filter(lambda col: "static_" in col, final_cols))
            final_static_cols=list(map(lambda col: "_".join(col.split("_")[1:]), final_static_cols))
            final_cols=list(filter(lambda col: "static_" not in col, final_cols))
            X_df=X_df[final_cols]
            static_df=static_df[final_static_cols]
            true_labels=df_label_valid["Label_WorseStateFromZero0.0To8.0Hours"]
            assert(true_labels.shape[0]==X_df.shape[0])
            X_feats=X_df[feat_order]
            X_full_collect=[X_feats]

            static_cols_without_encode_final = list(filter(lambda col2: col2 in static_cols_without_encode,
                                                           static_df.columns.values.tolist()))
            static_cols_one_hot_encode_final = list(filter(lambda col2: col2 in static_cols_one_hot_encode,
                                                           static_df.columns.values.tolist()))
            static_cols_one_hot_encode_str_final = list(filter(lambda col2: col2 in static_cols_one_hot_encode_str,
                                                               static_df.columns.values.tolist()))

            for col in static_cols_without_encode_final:
                static_val=float(static_df[col].iloc[0])
                col_arr=mlhc_array.value_empty((X_feats.shape[0],1),static_val)
                X_full_collect.append(col_arr)

            for col in static_cols_one_hot_encode_final:
                static_val=int(static_df[col].iloc[0])
                col_arr=mlhc_array.value_empty((X_feats.shape[0],1),static_val)
                X_full_collect.append(col_arr)

            for col in static_cols_one_hot_encode_str_final:
                static_val=str_to_int_dict[str(static_df[col].iloc[0])]
                col_arr=mlhc_array.value_empty((X_feats.shape[0],1),static_val)
                X_full_collect.append(col_arr)

            X_full=np.concatenate(X_full_collect,axis=1)
            
            pred_scores=orig_model.predict_proba(X_full)[:,1]

            if configs["only_alarm_points"]:
                true_labels=true_labels[pred_scores>critical_score] 
                pred_scores=pred_scores[pred_scores>critical_score]
            
            cum_cal_scores.append(pred_scores)
            cum_cal_labels.append(true_labels)

            n_valid_count+=1

        cal_scores_dict[(ml_model,col_desc,split_key,mimic_split_key)]=np.concatenate(cum_cal_scores)
        cal_labels_dict[(ml_model,col_desc,split_key,mimic_split_key)]=np.concatenate(cum_cal_labels)

        print("Number of processed val PIDs: {}/{}".format(n_valid_count,len(calibration_pids)))

    split_dict={"shap_top500_features": [("held_out",None),("temporal_1",None),("temporal_2",None),
                                         ("temporal_3",None),("temporal_4",None),("temporal_5",None)],
                "shap_top20_variables_MIMIC_BERN": [("held_out",None),("temporal_1",None),("temporal_2",None),
                                                    ("temporal_3",None),("temporal_4",None),("temporal_5",None)],
                "shap_top20_variables_MIMIC": [("held_out","random_0"),
                                               ("held_out","random_1"),("held_out","random_2"),
                                               ("held_out","random_3"),("held_out","random_4")],
                "shap_top20_variables_MIMIConly": [("held_out","random_0"),("held_out","random_1"),("held_out","random_2"),
                                                   ("held_out","random_3"),("held_out","random_4")]}

    label_dict={"shap_top500_features": "Full", "shap_top20_variables_MIMIC_BERN": "Compact", "shap_top20_variables_MIMIC": "MIMICval", "shap_top20_variables_MIMIConly": "MIMICretrain"}
    plot_title_dict={"shap_top20_variables_MIMIC": "circEWS-lite (MIMICval), alarm system"}

    MODELS_TO_ANALYZE=["shap_top20_variables_MIMIC_BERN","shap_top500_features"]
    
    for model_desc in MODELS_TO_ANALYZE:
        print("Building plot for model: {}".format(model_desc))

        # Summary-metrics
        original_bs=[]
        calibrated_bs_platt=[]
        calibrated_bs_iso=[]

        original_gini=[]
        original_gini_norm=[]
        iso_gini=[]
        iso_gini_norm=[]

        # Error bars
        mean_preds_orig=[]
        mean_preds_platt=[]
        mean_preds_iso=[]

        # References axes to interpolate to
        ref_axis_orig=None
        ref_axis_platt=None
        ref_axis_iso=None
        
        for split_key,mimic_split_key in split_dict[model_desc]:
            print("Processing split: {}".format(split_key))

            if "MIMIConly" in model_desc:
                model_col_desc=model_desc+"_"+mimic_split_key
            else:
                model_col_desc=model_desc
                
            labels_split=labels_dict[("lightgbm",model_col_desc,split_key,mimic_split_key)]
            scores_split=scores_dict[("lightgbm",model_col_desc,split_key,mimic_split_key)]
            labels_cal_split=cal_labels_dict[("lightgbm",model_col_desc,split_key,mimic_split_key)]
            scores_cal_split=cal_scores_dict[("lightgbm",model_col_desc,split_key,mimic_split_key)]

            if len(np.unique(labels_split))==1:
                labels_split[0]=1.0
            if len(np.unique(labels_cal_split))==1:
                labels_cal_split[0]=1.0

            cal_model=sklm.LogisticRegression()
            cal_model.fit(np.expand_dims(scores_cal_split,1), labels_cal_split)
            scores_calibrated_train=cal_model.predict_proba(np.expand_dims(scores_cal_split,1))[:,1]
            min_scale=scores_calibrated_train.min()
            max_scale=scores_calibrated_train.max()
            scores_calibrated=cal_model.predict_proba(np.expand_dims(scores_split,1))[:,1]
            scores_calibrated=(scores_calibrated-min_scale)/(max_scale-min_scale)
            scores_calibrated[scores_calibrated<0.0]=0.0
            scores_calibrated[scores_calibrated>1.0]=1.0

            cal_model=skiso.IsotonicRegression(out_of_bounds="clip",y_min=0.0,y_max=1.0)
            cal_model.fit(scores_cal_split, labels_cal_split)
            scores_calibrated_iso=cal_model.predict(scores_split)

            if configs["prev_corrected"]:
                split_prevalence=np.sum(labels_split==1.0)/labels_split.size
                hirid_labels=labels_dict[("lightgbm","shap_top20_variables_MIMIC_BERN","held_out",None)]
                hirid_prevalence=np.sum(hirid_labels==1.0)/hirid_labels.size
                local_correct_factor=(1-hirid_prevalence)*split_prevalence / ( hirid_prevalence * (1-split_prevalence) )                
                frac_pos_orig,mean_pred_orig=rescaled_calibration_curve(labels_split,scores_split,n_bins=20,correct_factor=local_correct_factor)
                frac_pos_calibrated,mean_pred_calibrated=rescaled_calibration_curve(labels_split,scores_calibrated,n_bins=20,correct_factor=local_correct_factor)
                frac_pos_calibrated_iso,mean_pred_calibrated_iso=rescaled_calibration_curve(labels_split,scores_calibrated_iso,n_bins=20,correct_factor=local_correct_factor)
            else:                
                frac_pos_orig,mean_pred_orig=rescaled_calibration_curve(labels_split,scores_split,n_bins=20,correct_factor=1.0)
                frac_pos_calibrated,mean_pred_calibrated=rescaled_calibration_curve(labels_split,scores_calibrated,n_bins=20,correct_factor=1.0)
                frac_pos_calibrated_iso,mean_pred_calibrated_iso=rescaled_calibration_curve(labels_split,scores_calibrated_iso,n_bins=20,correct_factor=1.0)

            # Raw scores
            try:
                frac_pos_orig_rs=scipy.interp(np.arange(0.0,1.01,0.01),mean_pred_orig,frac_pos_orig)
            except:
                continue
                
            ideal_diag=np.arange(0.0,1.01,0.01)
            diff_curve=np.absolute(frac_pos_orig_rs-ideal_diag)
            gini_coeff=skmetrics.auc(np.arange(0.0,1.01,0.01),diff_curve)
            original_gini.append(gini_coeff)

            diff_curve_norm=np.absolute(frac_pos_orig-mean_pred_orig)
            
            try:
                gini_coeff_norm=skmetrics.auc(mean_pred_orig,diff_curve_norm)/(mean_pred_orig.max()-mean_pred_orig.min())
            except:
                continue
                
            original_gini_norm.append(gini_coeff_norm)

            # Calibrated scores
            try:
                frac_pos_iso_rs=scipy.interp(np.arange(0.0,1.01,0.01),mean_pred_calibrated_iso,frac_pos_calibrated_iso)
            except:
                continue
                
            ideal_diag=np.arange(0.0,1.01,0.01)
            diff_curve=np.absolute(frac_pos_iso_rs-ideal_diag)
            gini_coeff=skmetrics.auc(np.arange(0.0,1.01,0.01),diff_curve)
            iso_gini.append(gini_coeff)

            diff_curve_norm=np.absolute(frac_pos_calibrated_iso-mean_pred_calibrated_iso)

            try:
                gini_coeff_norm=skmetrics.auc(mean_pred_calibrated_iso,diff_curve_norm)/(mean_pred_calibrated_iso.max()-mean_pred_calibrated_iso.min())
            except:
                continue
                
            iso_gini_norm.append(gini_coeff_norm)
            
            if ref_axis_orig is None:
                ref_axis_orig=mean_pred_orig
                ref_axis_platt=mean_pred_calibrated
                ref_axis_iso=mean_pred_calibrated_iso

            mean_preds_orig.append(scipy.interp(ref_axis_orig,mean_pred_orig,frac_pos_orig))
            mean_preds_platt.append(scipy.interp(ref_axis_platt,mean_pred_calibrated,frac_pos_calibrated))
            mean_preds_iso.append(scipy.interp(ref_axis_iso,mean_pred_calibrated_iso,frac_pos_calibrated_iso))

            bscore_orig=skmetrics.brier_score_loss(labels_split,scores_split)
            bscore_calibrated=skmetrics.brier_score_loss(labels_split,scores_calibrated)
            bscore_calibrated_iso=skmetrics.brier_score_loss(labels_split,scores_calibrated_iso)

            original_bs.append(bscore_orig)
            calibrated_bs_platt.append(bscore_calibrated)
            calibrated_bs_iso.append(bscore_calibrated_iso)

        data_out_dict={}

        data_out_dict["perfect_cal_x"]=[0,1]
        data_out_dict["perfect_cal_y"]=[0,1]

        data_out_dict["raw_x"]=ref_axis_orig
        data_out_dict["raw_y"]=np.mean(mean_preds_orig,axis=0)

        data_out_dict["raw_brier_mean"]=np.mean(original_bs)
        data_out_dict["raw_brier_std"]=np.std(original_bs)

        data_out_dict["raw_gini_mean"]=np.mean(original_gini)
        data_out_dict["raw_gini_std"]=np.std(original_gini)
        data_out_dict["raw_gini_norm_mean"]=np.mean(original_gini_norm)
        data_out_dict["raw_gini_norm_std"]=np.std(original_gini_norm)

        data_out_dict["raw_fill_min"]=np.maximum(np.mean(mean_preds_orig,axis=0)-np.std(mean_preds_orig,axis=0),0)
        data_out_dict["raw_fill_max"]=np.minimum(np.mean(mean_preds_orig,axis=0)+np.std(mean_preds_orig,axis=0),1)
        
        data_out_dict["iso_x"]=ref_axis_iso
        data_out_dict["iso_y"]=np.mean(mean_preds_iso,axis=0)

        data_out_dict["iso_brier_mean"]=np.mean(calibrated_bs_iso)
        data_out_dict["iso_brier_std"]=np.std(calibrated_bs_iso)

        data_out_dict["iso_gini_mean"]=np.mean(iso_gini)
        data_out_dict["iso_gini_std"]=np.std(iso_gini)
        data_out_dict["iso_gini_norm_mean"]=np.mean(iso_gini_norm)
        data_out_dict["iso_gini_norm_std"]=np.std(iso_gini_norm)

        data_out_dict["iso_fill_min"]=np.maximum(np.mean(mean_preds_iso,axis=0)-np.std(mean_preds_iso,axis=0),0)
        data_out_dict["iso_fill_max"]=np.minimum(np.mean(mean_preds_iso,axis=0)+np.std(mean_preds_iso,axis=0),1)
        data_out_dict["num_test_pids"]=num_filtered_pids

        if not configs["cohort_category_fine"]=="global":
            mlhc_io.save_pickle(data_out_dict,os.path.join(configs["data_out_dir"],"{}_{}_{}_{}_{}_{}_{}.pickle".format(model_desc,configs["cohort_category"], configs["cohort_value"].replace("/","-"),configs["cohort_category_fine"],configs["cohort_value_fine"],
                                                                                                                  "alarm" if configs["only_alarm_points"] else "score", "corrected" if configs["prev_corrected"] else "uncorrected")))
        else:
            mlhc_io.save_pickle(data_out_dict,os.path.join(configs["data_out_dir"],"{}_{}_{}_{}_{}.pickle".format(model_desc,configs["cohort_category"], configs["cohort_value"].replace("/","-"),
                                                                                                                  "alarm" if configs["only_alarm_points"] else "score", "corrected" if configs["prev_corrected"] else "uncorrected")))
        
def parse_cmd_args():
    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--predictions_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/8_predictions/181108", help="Which predictions to analyze?")
    
    parser.add_argument("--bern_pid_map_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/id_lists/v6b/patients_in_clean_chunking_50.pickle", help="Path of the PID map")
    parser.add_argument("--mimic_pid_map_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/misc_derived/id_lists/chunks_181003.pickle", help="Path of MIMIC PID map")

    parser.add_argument("--shapley_values_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/8_predictions/181108/reduced", help="Absolute shapley values computed on the features")
    
    parser.add_argument("--temporal_split_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/temporal_split_180918.pickle", help="Path of temporal split descriptor")
    parser.add_argument("--mimic_split_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/misc_derived/split_181015.pickle",
                        help="Path of MIMIC split descriptor")
    
    parser.add_argument("--bern_ml_input_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/7_ml_input/180918", help="ML input dir for HIRID")
    parser.add_argument("--bern_imputed_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5_imputed/imputed_180918", help="Imputation directory to use for the HIRID")
    parser.add_argument("--mimic_ml_input_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/ml_input/181023", help="ML input dir of MIMIC")
    parser.add_argument("--mimic_imputed_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/imputed/imputed_181023", help="Imputed dir of MIMIC")
    parser.add_argument("--threshold_map", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/circews_analysis/alarm_score_for_calibration/v6b/threshold_info.csv",
                        help="Dictionary to look up the alarm system thresholds")
    
    parser.add_argument("--shapelets_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/9_s3m/feature_matrices_fixed", help="Feature matrix file containing the shapelets")
    parser.add_argument("--alarms_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/circews_analysis/alarm_score_for_calibration/v6b")
    parser.add_argument("--pid_category_map", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/pid_patient_subcohort_mapping.tsv", help="PID category map for cohort analyses")

    # Output paths
    parser.add_argument("--plot_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/plots/mhueser/181108", help="Plotting base path")
    parser.add_argument("--data_out_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/model_calibration", help="Path where to store intermediate data")

    # Arguments
    parser.add_argument("--data_mode", default="reduced", help="Should reduced data be used?")
    parser.add_argument("--task_key", default="WorseStateFromZero", help="Which label should be evaluated?")
    parser.add_argument("--lhours", default=0.0, type=float, help="Left boundary of future horizon")
    parser.add_argument("--rhours", default=8.0, type=float, help="Right boundary of future horizon")
    parser.add_argument("--val_type", default="test", help="Which data set to evaluate with?")
    parser.add_argument("--calibration_set", default="val", help="Which data-set should be used for calibration of model?")
    parser.add_argument("--verbose", default=False, help="Should verbose messages be printed?")
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debug mode with fewer patients")
    parser.add_argument("--random_state", type=int, default=2019, help="Random state of RNG")
    parser.add_argument("--only_alarm_points", default=False, action="store_true", help="Should only alarms be analyzed?")
    parser.add_argument("--desired_recall", default=0.9,type=float,help="Recall at which the analysis is performed")
    parser.add_argument("--prev_corrected", default=False, action="store_true", help="Prevalence correction applied")
    parser.add_argument("--cohort_category", default="global", help="Cohort category to filter test set PIDs for")
    parser.add_argument("--cohort_category_fine", default="global", help="Cohort category to filter test set PIDs for")    
    parser.add_argument("--cohort_value", default="all", help="Cohort sub-group to validate")
    parser.add_argument("--cohort_value_fine", default="all", help="Cohort sub-group to validate")

    configs=vars(parser.parse_args())
    return configs
    

if __name__=="__main__":
    configs=parse_cmd_args()
    calibration_metrics(configs)
