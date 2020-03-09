
import os
import os.path
import csv
import sys

def translate_horizons(feat_name):
    ''' Translates horizons'''
    if "vm1_" in feat_name and "high" in feat_name:
        return feat_name.replace("high_","med_")
    elif "vm150_" in feat_name and "med" in feat_name:
        return feat_name.replace("med_","low_")
    elif "vm58_" in feat_name and "high" in feat_name:
        return feat_name.replace("high_","med_")
    elif "vm13_" in feat_name and "high" in feat_name:
        return feat_name.replace("high_","med_")
    elif "vm149_" in feat_name and "med" in feat_name:
        return feat_name.replace("med_","low_")
    elif "vm135_" in feat_name and "med" in feat_name:
        return feat_name.replace("med_","low_")
    elif "vm61_" in feat_name and "high" in feat_name:
        return feat_name.replace("high_","med_")
    elif "vm3_" in feat_name and "high" in feat_name:
        return feat_name.replace("high_","med_")
    elif "vm5_" in feat_name and "high" in feat_name:
        return feat_name.replace("high_","med_")
    elif "vm4_" in feat_name and "high" in feat_name:
        return feat_name.replace("high_","med_")
    elif "vm20_" in feat_name and "high" in feat_name:
        return feat_name.replace("high_","med_")
    elif "vm65_" in feat_name and "high" in feat_name:
        return feat_name.replace("high_","med_")
    elif "vm29_" in feat_name and "high" in feat_name:
        return feat_name.replace("high_","med_")
    elif "vm33_" in feat_name and "high" in feat_name:
        return feat_name.replace("high_","med_")
    else:
        return feat_name

def yield_final_cols(column_desc, sel_cols_X=None, configs=None):

    shapley_values_path=os.path.join(configs["shapley_values_path"],configs["split_key"],
                                     "WorseStateFromZero_0.0_8.0_normal_model_shapelets_50_percent_lightgbm_full",
                                     "best_model_shapley_values.tsv")
    
    if "normal_model_neg_subsample" in column_desc or column_desc=="normal_model_20_percent" or column_desc=="normal_model_50_percent" \
       or column_desc=="normal_model_shapelets_50_percent" or column_desc=="normal_model_shapelets_50_percent_lowres" \
       or column_desc=="normal_model_regularization" or column_desc=="without_datetime": 
        final_cols=sel_cols_X

    # SHAPLEY VALUE BASED TOP FEATURES

    # Top 1000 features from Shapley values
    
    elif column_desc in ["shap_top1000_features"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        final_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-1000:]
        
    elif column_desc in ["shap_top500_features","0p1_percent_data", "1_percent_data", "5_percent_data", "10_percent_data","20_percent_data","50_percent_data",
                         "ml_compare_tree", "ml_compare_logreg", "ml_compare_mlp",
                         "ml_compare_tree_50", "ml_compare_logreg_50", "ml_compare_mlp_50",
                         "ml_compare_tree_100", "ml_compare_logreg_100", "ml_compare_mlp_100","dummy_output","downsampled_hirid_full_orig"] \
         or "tempgen_" in column_desc:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
                csv_fp=csv.reader(fp,delimiter='\t')
                next(csv_fp)
                for fname,score in csv_fp:
                    tuple_lst.append((fname,float(score)))
        final_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]

    elif column_desc in ["downsampled_hirid_full"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
                csv_fp=csv.reader(fp,delimiter='\t')
                next(csv_fp)
                for fname,score in csv_fp:
                    tuple_lst.append((fname,float(score)))
        final_cols=list(map(lambda fname: translate_horizons(fname), list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]))
        final_cols=list(filter(lambda fname: "dist-set" not in fname, final_cols))

    # Top 200 features from Shapley values
    elif column_desc=="shap_top200_features":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        final_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-200:]

    # Top 100 features from Shapley values
    elif column_desc=="shap_top100_features":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        final_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-100:]

    # Top 50 features from Shapley values
    elif column_desc=="shap_top50_features":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        final_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-50:]

    # Top 20 features from Shapley values
    elif column_desc=="shap_top20_features":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        final_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-20:]

    elif column_desc=="shap_nonzero_features":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        final_cols=list(map(lambda elem: elem[0], filter(lambda tpl: tpl[1]>0.0, tuple_lst)))

        
    # HORIZON LENGTH EXPERIMENT

    elif column_desc=="short_term_horizon":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "ratio_0" in col or "high_0" in col or "med_0" in col or "low_0" in col, filter_cols))

    elif column_desc=="plus_med_term_horizon":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "ratio_0" in col or "high_0" in col or "med_0" in col or "low_0" in col \
                               or "ratio_1" in col or "high_1" in col or "med_1" in col or "low_1" in col, filter_cols))


    elif column_desc=="plus_longer_term_horizon":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "ratio_0" in col or "high_0" in col or "med_0" in col or "low_0" in col \
                               or "ratio_1" in col or "high_1" in col or "med_1" in col or "low_1" in col \
                               or "ratio_2" in col or "high_2" in col or "med_2" in col or "low_2" in col, filter_cols))


    elif column_desc=="plus_longest_term_horizon":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "ratio_0" in col or "high_0" in col or "med_0" in col or "low_0" in col \
                               or "ratio_1" in col or "high_1" in col or "med_1" in col or "low_1" in col \
                               or "ratio_2" in col or "high_2" in col or "med_2" in col or "low_2" in col \
                               or "ratio_3" in col or "high_3" in col or "med_3" in col or "low_3" in col, filter_cols))

    elif column_desc=="only_longest_term_horizon":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "ratio_1" in col or "high_1" in col or "med_1" in col or "low_1" in col, filter_cols))
    
    # SUMMARY FUNCTION EXPERIMENT

    elif column_desc=="summary_location":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "_mean" in col or "_median" in col, filter_cols))

    elif column_desc=="summary_location_trend":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "_mean" in col or "_median" in col or "_trend" in col , filter_cols))

    elif column_desc=="summary_all":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "_mean" in col or "_median" in col or "_trend" in col \
                               or "_max" in col or "_min" in col or "_std" in col or "_iqr" in col, filter_cols))


    # SPECIFIED CLINICAL BASELINES

    elif column_desc=="clinical_baseline_map_lactate_hr_age":  
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col or "vm146_" in col or "vm1_" in col \
                               or col in ["plain_vm136", "plain_vm146", "plain_vm5","plain_vm1","static_Age"] \
                               or "lac_" in col or "map_" in col, filter_cols))


    elif column_desc=="clinical_baseline_only_vitals":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm5_" in col or "vm1_" in col or "vm3_" in col or "vm4_" in col \
                               or "vm20_" in col or "vm28_" in col 
                               or col in ["plain_vm5","plain_vm1", "plain_vm3","plain_vm4","plain_vm20", "plain_vm28", "RelDatetime","static_Age"] \
                               or "map_" in col, filter_cols))

    elif column_desc=="lactate_baseline":  
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm146_" in col \
                               or col in ["plain_vm136", "plain_vm146"] \
                               or "lac_" in col, filter_cols))

    elif column_desc=="lactate_map_baseline":  
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col or "vm146_" in col \
                               or col in ["plain_vm136", "plain_vm146", "plain_vm5"] \
                               or "lac_" in col or "map_" in col, filter_cols))

    elif column_desc in ["only_current_simple_dtree"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: col in ["plain_vm136", "plain_vm146", "plain_vm5","RelDatetime"], filter_cols))

    # SHAPLEY TOP VARIABLES PREFIX EXPERIMENTS
        
    elif column_desc=="shap_top1_variables": 
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or col in ["plain_vm136"], filter_cols))


    elif column_desc=="shap_top2_variables":  
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm146_" in col \
                               or col in ["plain_vm136", "plain_vm146"] \
                               or "lac_" in col, filter_cols))

    elif column_desc=="shap_top3_variables": 
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col or "vm146_" in col \
                               or col in ["plain_vm136", "plain_vm5","plain_vm146"] \
                               or "map_" in col or "lac_" in col, filter_cols))

    elif column_desc=="shap_top4_variables": 
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col or "vm146_" in col \
                               or col in ["plain_vm136", "plain_vm5", "RelDatetime","plain_vm146"] \
                               or "map_" in col or "lac_" in col, filter_cols))

    elif column_desc=="shap_top5_variables":  
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col or "vm146_" in col \
                               or col in ["plain_vm136", "plain_vm146", "plain_vm5", "RelDatetime","static_Age"] \
                               or "lac_" in col or "map_" in col, filter_cols))

    elif column_desc=="shap_top10_variables":  
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm146_" in col or "vm5_" in col \
                               or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                               or col in ["plain_vm136", "plain_vm146", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43",
                                          "plain_pm44", "plain_vm1", "RelDatetime","static_Age"] \
                               or "map_" in col or "lac_" in col or "dop_" in col \
                               or "mil_" in col or "lev_" in col or "theo_" in col \
                               or "event1_" in col, filter_cols))

    elif column_desc=="shap_top15_variables": 
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm146_" in col or "vm5_" in col \
                               or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                               or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                               or "vm176_" in col \
                               or col in ["plain_vm136", "plain_vm146", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                          "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176", "RelDatetime","static_Age"] \
                               or "map_" in col or "lac_" in col or "dop_" in col \
                               or "mil_" in col or "lev_" in col or "theo_" in col \
                               or "event1_" in col, filter_cols))

    elif column_desc=="shap_top20_variables":  
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        im_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm146_" in col or "vm5_" in col \
                               or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                               or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                               or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col or "pm87_" in col \
                               or col in ["plain_vm136", "plain_vm146", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                          "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176", "RelDatetime","static_Age",
                                          "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20", "plain_pm87"] \
                               or "map_" in col or "lac_" in col or "dop_" in col \
                               or "mil_" in col or "lev_" in col or "theo_" in col \
                               or "event1_" in col, im_cols))


    elif column_desc in ["shap_top20_variables_MIMIC","shap_top20_variables_MIMIConly",
                         "shap_top20_variables_MIMIConly_random_0",
                         "shap_top20_variables_MIMIConly_random_1",
                         "shap_top20_variables_MIMIConly_random_2",
                         "shap_top20_variables_MIMIConly_random_3",
                         "shap_top20_variables_MIMIConly_random_4",
                         "shap_top20_variables_MIMIConly_SS_random_0",
                         "shap_top20_variables_MIMIConly_SS_random_1",
                         "shap_top20_variables_MIMIConly_SS_random_2",
                         "shap_top20_variables_MIMIConly_SS_random_3",
                         "shap_top20_variables_MIMIConly_SS_random_4",
                         "shap_top20_variables_MIMIConly_held_out","mimic_dummy_output",
                         "shap_top20_variables_MIMIC_BERN","circewslite_top18_downsampling",
                         "circewslite_binaryind_downsampled"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col \
                               or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                               or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                               or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col \
                               or col in ["plain_vm136", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                          "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176", "RelDatetime","static_Age",
                                          "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20"] \
                               or "map_" in col or "lac_" in col or "dop_" in col \
                               or "mil_" in col or "lev_" in col or "theo_" in col \
                               or "event1_" in col, filter_cols))

    elif column_desc in ["circewslite_binaryind_v2", "circewslite_upsampled_downsampled", "circewslite_train_usds_test_orig","circewslite_train_orig_test_usds"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        filter_cols=list(filter(lambda fname: "dist-set" not in fname, filter_cols)) # HAVE TO REMOVE SHAPELETS AS THEY CAUSE ISSUES        
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col \
                               or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                               or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                               or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col \
                               or col in ["plain_vm136", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                          "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176", "RelDatetime","static_Age",
                                          "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20"] \
                               or "map_" in col or "lac_" in col or "dop_" in col \
                               or "mil_" in col or "lev_" in col or "theo_" in col \
                               or "event1_" in col, filter_cols))

    elif column_desc in ["circewslite_upsampled_downsampled_with_shapelets_redisc","circewslite_upsampled_downsampled_with_shapelets_applied",
                         "circewslite_train_orig_test_usds_with_shapelets"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col \
                               or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                               or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                               or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col \
                               or col in ["plain_vm136", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                          "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176", "RelDatetime","static_Age",
                                          "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20"] \
                               or "map_" in col or "lac_" in col or "dop_" in col \
                               or "mil_" in col or "lev_" in col or "theo_" in col \
                               or "event1_" in col, filter_cols))

    elif column_desc in ["circewslite_train_binaryind_test_binaryind_usds_no_shapelets","circewslite_train_binaryind_test_binaryind_usds_rev2_no_shapelets",
                         "circewslite_train_binaryind_usds_test_binaryind_usds_rev2_no_shapelets"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        filter_cols=list(filter(lambda fname: "dist-set" not in fname, filter_cols)) # HAVE TO REMOVE SHAPELETS AS THEY CAUSE ISSUES
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col \
                               or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                               or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                               or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col \
                               or col in ["plain_vm136", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                          "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176",
                                          "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20"] \
                               or "map_" in col or "lac_" in col or "dop_" in col \
                               or "mil_" in col or "lev_" in col or "theo_" in col \
                               or "event1_" in col, filter_cols))
        
    elif column_desc in ["circewslite_binaryind_with_shapelets_redisc","circewslite_train_binaryind_test_binaryind_usds"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col \
                               or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                               or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                               or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col \
                               or col in ["plain_vm136", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                          "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176",
                                          "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20"] \
                               or "map_" in col or "lac_" in col or "dop_" in col \
                               or "mil_" in col or "lev_" in col or "theo_" in col \
                               or "event1_" in col, filter_cols))

    elif column_desc in ["circewslite_no_drugs"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col \
                               or "vm1_" in col or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                               or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col \
                               or col in ["plain_vm136", "plain_vm5","plain_vm1","plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176",
                                          "RelDatetime","static_Age","plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20"],filter_cols))

    elif column_desc in ["circewslite_binarized_upsampled_downsampled_no_drugs","circewslite_upsampled_downsampled_no_drugs"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        filter_cols=list(filter(lambda fname: "dist-set" not in fname, filter_cols)) # HAVE TO REMOVE SHAPELETS AS THEY CAUSE ISSUES
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col \
                               or "vm1_" in col or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                               or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col \
                               or col in ["plain_vm136", "plain_vm5","plain_vm1","plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176",
                                          "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20"],filter_cols))

    elif column_desc in ["only_reldatetime"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: col in ["RelDatetime"],filter_cols))
        
    elif column_desc in ["downsampled_hirid_compact","ds_hirid_compact_MIMIC","downsampled_hirid_compact_orig_no_shapelets","shap_top20_variables_MIMIC_BERN_no_shapelets",
                         "alt_endpoint_MAP60","alt_endpoint_MAP67","impute_only_forward_filling","impute_no_imputation","circewslite_top18_downsampling_no_shapelets",
                         "circewslite_binaryind_no_shapelets_v2", "circewslite_binaryind_downsampled_no_shapelets"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        filter_cols=list(filter(lambda fname: "dist-set" not in fname, filter_cols))
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col \
                               or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                               or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                               or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col \
                               or col in ["plain_vm136", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                          "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176", "RelDatetime","static_Age",
                                          "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20"] \
                               or "map_" in col or "lac_" in col or "dop_" in col \
                               or "mil_" in col or "lev_" in col or "theo_" in col \
                               or "event1_" in col, filter_cols))

    elif column_desc in ["circewslite_binaryind_no_shapelets_no_datetime", "circewslite_binaryind_downsampled_no_shapelets_no_datetime"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        filter_cols=list(filter(lambda fname: "dist-set" not in fname, filter_cols))
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col \
                               or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                               or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                               or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col \
                               or col in ["plain_vm136", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                          "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176","plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20"] \
                               or "map_" in col or "lac_" in col or "dop_" in col \
                               or "mil_" in col or "lev_" in col or "theo_" in col \
                               or "event1_" in col, filter_cols))

    elif column_desc in ["circewslite_binaryind_no_shapelets_only_pharma"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        filter_cols=list(filter(lambda fname: "dist-set" not in fname, filter_cols))
        im_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col \
                             or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                             or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                             or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col \
                             or col in ["plain_vm136", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                        "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176", "RelDatetime","static_Age",
                                        "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20"] \
                             or "map_" in col or "lac_" in col or "dop_" in col \
                             or "mil_" in col or "lev_" in col or "theo_" in col \
                             or "event1_" in col, filter_cols))        
        final_cols=list(filter(lambda col: "pm" in col or "dop_" in col or "mil_" in col or "lev_" in col or "theo_" in col \
                               or col in ["RelDatetime", "static_Age"], im_cols))

    elif column_desc in ["circewslite_binaryind_only_pharma"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        im_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col \
                             or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                             or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                             or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col \
                             or col in ["plain_vm136", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                        "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176", "RelDatetime","static_Age",
                                        "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20"] \
                             or "map_" in col or "lac_" in col or "dop_" in col \
                             or "mil_" in col or "lev_" in col or "theo_" in col \
                             or "event1_" in col, filter_cols))        
        final_cols=list(filter(lambda col: "pm" in col or "dop_" in col or "mil_" in col or "lev_" in col or "theo_" in col \
                               or col in ["RelDatetime", "static_Age"], im_cols))

    elif column_desc in ["circewslite_binaryind_no_shapelets_only_measurements"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        filter_cols=list(filter(lambda fname: "dist-set" not in fname, filter_cols))
        im_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col \
                             or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                             or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                             or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col \
                             or col in ["plain_vm136", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                        "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176", "RelDatetime","static_Age",
                                        "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20"] \
                             or "map_" in col or "lac_" in col or "dop_" in col \
                             or "mil_" in col or "lev_" in col or "theo_" in col \
                             or "event1_" in col, filter_cols))        
        final_cols=list(filter(lambda col: "vm" in col or "map_" in col or "lac_" in col \
                               or col in ["RelDatetime", "static_Age"], im_cols))

    elif column_desc in ["circewslite_binaryind_only_measurements"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        im_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col \
                             or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                             or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                             or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col \
                             or col in ["plain_vm136", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                        "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176", "RelDatetime","static_Age",
                                        "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20"] \
                             or "map_" in col or "lac_" in col or "dop_" in col \
                             or "mil_" in col or "lev_" in col or "theo_" in col \
                             or "event1_" in col, filter_cols))        
        final_cols=list(filter(lambda col: "vm" in col or "map_" in col or "lac_" in col \
                               or col in ["RelDatetime", "static_Age"], im_cols))        

    elif column_desc in ["downsampled_hirid_compact_orig","ds_hirid_compact_MIMIC_orig","alt_endpoint_MAP60_with_shapelets",
                         "alt_endpoint_MAP67_with_shapelets","impute_only_forward_filling_with_shapelets","impute_no_imputation_with_shapelets"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col \
                               or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                               or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                               or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col \
                               or col in ["plain_vm136", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                          "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176", "RelDatetime","static_Age",
                                          "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20"] \
                               or "map_" in col or "lac_" in col or "dop_" in col \
                               or "mil_" in col or "lev_" in col or "theo_" in col \
                               or "event1_" in col, filter_cols))

    elif column_desc in ["ds_hirid_compact_MIMIConly_r0","ds_hirid_compact_MIMIConly_r1",
                         "ds_hirid_compact_MIMIConly_r2","ds_hirid_compact_MIMIConly_r3","ds_hirid_compact_MIMIConly_r4",
                         "ds_hirid_compact_MIMIConly_r0_orig","ds_hirid_compact_MIMIConly_r1_orig",
                         "ds_hirid_compact_MIMIConly_r2_orig","ds_hirid_compact_MIMIConly_r3_orig",
                         "ds_hirid_compact_MIMIConly_r4_orig","ds_hirid_compact_MIMIConly_r0_orig_no_shapelets",
                         "ds_hirid_compact_MIMIConly_r1_orig_no_shapelets","ds_hirid_compact_MIMIConly_r2_orig_no_shapelets",
                         "ds_hirid_compact_MIMIConly_r3_orig_no_shapelets","ds_hirid_compact_MIMIConly_r4_orig_no_shapelets"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        filter_cols=list(filter(lambda fname: "dist-set" not in fname, filter_cols))        
        final_cols=list(filter(lambda col: "vm136_" in col or "vm5_" in col \
                               or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                               or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                               or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col \
                               or col in ["plain_vm136", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                          "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176", "RelDatetime","static_Age",
                                          "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20"] \
                               or "map_" in col or "lac_" in col or "dop_" in col \
                               or "mil_" in col or "lev_" in col or "theo_" in col \
                               or "event1_" in col, filter_cols))

    elif column_desc=="shap_top30_variables": 
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm146_" in col or "vm5_" in col \
                               or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                               or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                               or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col or "pm87_" in col \
                               or "vm23_" in col or "vm132_" in col or "vm139_" in col or "vm63_" in col or "vm133_" in col or "vm65_" in col \
                               or "pm39_" in col or "vm58_" in col or "vm134_" in col or "vm168_" in col \
                               or col in ["plain_vm136", "plain_vm146", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                          "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176", "RelDatetime","static_Age",
                                          "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20", "plain_pm87",
                                          "plain_vm23", "plain_vm132", "plain_vm139", "plain_vm63", "plain_vm133", "plain_vm65",
                                          "plain_pm39", "plain_vm58", "plain_vm134", "plain_vm168"] \
                               or "map_" in col or "lac_" in col or "dop_" in col \
                               or "mil_" in col or "lev_" in col or "theo_" in col or "noreph_" in col \
                               or "event1_" in col, filter_cols))


    elif column_desc=="shap_top50_variables":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" in col or "vm146_" in col or "vm5_" in col \
                               or "pm41_" in col or "pm42_" in col or "pm43_" in col or "pm44_" in col or "vm1_" in col \
                               or "vm13_" in col or "vm28_" in col or "vm172_" in col or "vm174_" in col \
                               or "vm176_" in col or "vm4_" in col or "vm62_" in col or "vm3_" in col or "vm20_" in col or "pm87_" in col \
                               or "vm23_" in col or "vm132_" in col or "vm139_" in col or "vm63_" in col or "vm133_" in col or "vm65_" in col \
                               or "pm39_" in col or "vm58_" in col or "vm134_" in col or "vm168_" in col \
                               or "vm148_" in col or "vm16_" in col or "vm32_" in col or "vm151_" in col \
                               or "vm135_" in col or "vm64_" in col or "vm34_" in col or "vm137_" in col \
                               or "vm60_" in col or "vm66_" in col or "vm22_" in col or "vm141_" in col \
                               or "vm163_" in col or "vm185_" in col or "vm140_" in col or "vm61_" in col \
                               or "vm153_" in col or "vm145_" in col or "vm24_" in col or "pm91_" in col \
                               or col in ["plain_vm136", "plain_vm146", "plain_vm5","plain_pm41", "plain_pm42", "plain_pm43", "plain_pm44","plain_vm1",
                                          "plain_vm13", "plain_vm28", "plain_vm172", "plain_vm174", "plain_vm176", "RelDatetime","static_Age",
                                          "plain_vm4", "plain_vm62", "plain_vm3", "plain_vm20", "plain_pm87",
                                          "plain_vm23", "plain_vm132", "plain_vm139", "plain_vm63", "plain_vm133", "plain_vm65",
                                          "plain_pm39", "plain_vm58", "plain_vm134", "plain_vm168","plain_vm148",
                                          "plain_vm16", "plain_vm32", "plain_vm151", "plain_vm135", "plain_vm64",
                                          "plain_vm34", "plain_vm137", "plain_vm60", "plain_vm66", "plain_vm22",
                                          "plain_vm141", "plain_vm163", "plain_vm185", "plain_vm140", "plain_vm61",
                                          "plain_vm153", "plain_vm145", "plain_vm24", "plain_pm91"] \
                               or "map_" in col or "lac_" in col or "dop_" in col \
                               or "mil_" in col or "lev_" in col or "theo_" in col or "noreph_" in col \
                               or "event1_" in col, filter_cols))

    # MARGINAL REMOVALS FROM THE REFERENCE MODEL (TOP500 SHAPLEY FEATURES)

    # Remove static variables from the reference model
    elif column_desc=="remove_static":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "static_" not in col, filter_cols))

    # Remove measurement-based features from the reference model
    elif column_desc in ["remove_shapelets","development_split","development_split_1","development_split_2",
                         "development_split_3","development_split_4","development_split_5","lustre_benchmark"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "_dist-set" not in col, filter_cols))

    # Only shapelet-based features will remain in the model
    elif column_desc=="only_shapelets": 
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "_dist-set" in col, filter_cols))

    # Remove measurement-based features from the reference model
    elif column_desc=="remove_measurement":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "time_to_last_ms" not in col and "measure_density" not in col, filter_cols))

    # Remove entire-stay based features from the reference model
    elif column_desc=="remove_entire_stay":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "_entire" not in col, filter_cols))

    # Remove time-to-critical event based feaures from the reference model
    elif column_desc=="remove_time_to_critical":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "time_to_last_ms" in col or "_time" not in col, filter_cols))

    # Remove all-pseudo-endpoint based features from the reference model
    elif column_desc=="remove_pseudo_endpoint":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "event" not in col and "epineph" not in col and "noreph" not in col, filter_cols))

    # Remove multi-scale history based features from the reference model
    elif column_desc=="remove_multiscale_history":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "RelDatetime" in col or "plain_" in col \
                               or "event" in col or "epineph" in col or "noreph" in col or "time_to_last_ms" in col \
                               or "measure_density" in col or "static_" in col, filter_cols))

    # Consider only the current patient state
    elif column_desc in ["only_current"]:
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "RelDatetime" in col or "plain_" in col or "static_" in col, filter_cols))

    # Univariate variable removal experiment for Thomas
        
    elif column_desc=="marginal_remove_lactate":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm136_" not in col and "vm146_" not in col \
                               and col not in ["plain_vm136", "plain_vm146"] \
                               and "lac_" not in col and "event1_" not in col \
                               and "event2_" not in col and "event3_" not in col, filter_cols))

    elif column_desc=="marginal_remove_map":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm5_" not in col \
                               and col not in ["plain_vm5"] \
                               and "map_" not in col and "event1_" not in col, filter_cols))

    elif column_desc=="marginal_remove_datetime":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: col not in ["RelDatetime"], filter_cols))

    elif column_desc=="marginal_remove_age":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: col not in ["static_Age"], filter_cols))

    elif column_desc=="marginal_remove_hr":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm1_" not in col \
                               and col not in ["plain_vm1"], filter_cols))

    elif column_desc=="marginal_remove_l1_drugs":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "pm41_" not in col and "pm42_" not in col \
                               and "pm43_" not in col and "pm44_" not in col \
                               and col not in ["plain_pm41","plain_pm42", "plain_pm43", "plain_pm44"] \
                               and "dop_" not in col and "mil_" not in col and "lev_" not in col and "theo_" not in col \
                               and "event1" not in col, filter_cols))

    elif column_desc=="marginal_remove_cardiac_output":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm13_" not in col \
                               and col not in ["plain_vm13"], filter_cols))


    elif column_desc=="marginal_remove_RASS":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm28_" not in col \
                               and col not in ["plain_vm28"], filter_cols))

    elif column_desc=="marginal_remove_INR":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm172_" not in col \
                               and col not in ["plain_vm172"], filter_cols))

    elif column_desc=="marginal_remove_glucose":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm174_" not in col \
                               and col not in ["plain_vm174"], filter_cols))

    elif column_desc=="marginal_remove_c_reactive_protein":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm176_" not in col \
                               and col not in ["plain_vm176"], filter_cols))
        

    elif column_desc=="marginal_remove_abpd":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm4_" not in col \
                               and col not in ["plain_vm4"], filter_cols))

    elif column_desc=="marginal_remove_peakpressure":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm62_" not in col \
                               and col not in ["plain_vm62"], filter_cols))

        
    elif column_desc=="marginal_remove_abps":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm3_" not in col \
                               and col not in ["plain_vm3"], filter_cols))

        
    elif column_desc=="marginal_remove_spo2":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm20_" not in col \
                               and col not in ["plain_vm20"], filter_cols))

        
    elif column_desc=="marginal_remove_opiod":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "pm87_" not in col \
                               and col not in ["plain_pm87"], filter_cols))

    elif column_desc=="marginal_remove_suppox":
        tuple_lst=[]
        with open(shapley_values_path,'r') as fp:
            csv_fp=csv.reader(fp,delimiter='\t')
            next(csv_fp)
            for fname,score in csv_fp:
                tuple_lst.append((fname,float(score)))
        filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-500:]
        final_cols=list(filter(lambda col: "vm23_" not in col \
                               and col not in ["plain_vm23"], filter_cols))

        
    else:
        print("ERROR: Invalid feature description",flush=True)
        sys.exit(1)

    return final_cols


