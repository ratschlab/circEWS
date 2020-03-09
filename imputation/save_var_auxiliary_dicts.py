'''
Save dictionaries to file system which holds meta-information about variables and
encodings which is later used during adaptive imputation.
'''

import ipdb
import sys
import argparse

import pandas as pd

import circews.functions.util.io as mlhc_io
import circews.functions.util.math as mlhc_math

def save_var_auxiliary_dicts(configs):
    df_varref=pd.read_csv(configs["current_varref_table"], delimiter='\t',encoding="latin")
    normal_vars=list(map(int, mlhc_io.read_list_from_file(configs["var_list_path"])))
    normal_dict={}
    encoding_dict={}

    # Save information about original physiological variables
    for var in normal_vars:

        df_red=df_varref[df_varref["VariableID"]==var]

        # No auxiliary info for pharma variables
        if df_red["Type"].values[0]=="Pharma":
            continue

        normal_val=df_red["NormalValue"].values[0]

        # Variable types
        if int(var) in configs["categorical_vars"]:
            encoding_dict["v{}".format(var)]="categorical"
        elif int(var) in configs["ordinal_vars"]:
            encoding_dict["v{}".format(var)]="ordinal"
        elif int(var) in configs["binary_vars"]:
            encoding_dict["v{}".format(var)]="binary"
        else:
            encoding_dict["v{}".format(var)]="continuous"

        if mlhc_math.is_numeric(normal_val):
            normal_dict["v{}".format(var)]=float(normal_val)            
        else:
            
            # SPECIAL CASE OF A FORMULA
            if "height^" in normal_val:
                normal_dict["v{}".format(var)]="CUSTOM_FORMULA"

            # STRINGS WITH MEANING NO
            elif normal_val in ["none","no"]:
                normal_dict["v{}".format(var)]="NO"
                
            # SIMPLE FORMULA, FILLING IN WEIGHT
            elif "kg weight" in normal_val:
                normal_dict["v{}".format(var)]="CUSTOM_FORMULA"

            # Encode string category with the correct bit pattern
            elif normal_val in configs["category_str_to_num"]:
                normal_dict["v{}".format(var)]=configs["category_str_to_num"][normal_val]

            # UNKNOWN NORMAL VALUE
            else:
                ipdb.set_trace()
                print("ERROR: Could not determine normal value")
                sys.exit(1)

    df_labref=pd.read_csv(configs["current_labref_table"], delimiter='\t',encoding="latin")
    lab_vars=mlhc_io.read_list_from_file(configs["labvar_list_path"])

    # Save information about original lab variables
    for var in lab_vars:
        df_red=df_labref[df_labref["VariableID"]==int(var)]
        normal_val=df_red["NormalValue"].values[0]

        if int(var) in configs["categorical_vars"]:
            encoding_dict["v{}".format(var)]="categorical"
        elif int(var) in configs["ordinal_vars"]:
            encoding_dict["v{}".format(var)]="ordinal"
        elif int(var) in configs["binary_vars"]:
            encoding_dict["v{}".format(var)]="binary"
        else:
            encoding_dict["v{}".format(var)]="continuous"
        
        if mlhc_math.is_numeric(normal_val):
            normal_dict["v{}".format(var)]=float(normal_val)
        else:
            print("ERROR: Could not determine normal value")
            sys.exit(1)

    if not configs["debug_mode"]:
        mlhc_io.save_pickle(normal_dict, configs["normalval_map_path"])
        mlhc_io.save_pickle(encoding_dict, configs["varencoding_map_path"])

    meta_normal_vars=mlhc_io.read_list_from_file(configs["meta_var_list_path"])
    meta_lab_vars=mlhc_io.read_list_from_file(configs["meta_labvar_list_path"])
    meta_normal_dict={}
    meta_encoding_dict={}

    # Meta-variables (non-pharma, normal variables)
    for var in meta_normal_vars:
        df_red=df_varref[df_varref["MetaVariableID"]==int(var)]

        # Decide which kind of variable we are dealing with (2 fundamental types: Pharma and Normal Variable)
        if df_red["Type"].values[0]=="Pharma":
            var_prefix="p"
        else:
            var_prefix="v"

        normal_val=df_red["NormalValue"].unique()
        assert(normal_val.size==1)
        normal_val=normal_val[0]

        if mlhc_math.is_numeric(normal_val):
            meta_normal_dict["{}m{}".format(var_prefix,var)]=float(normal_val)
        elif "height^" in normal_val or "kg weight" in normal_val:
            meta_normal_dict["{}m{}".format(var_prefix,var)]="CUSTOM_FORMULA"
        elif normal_val in configs["category_str_to_num"]:
            meta_normal_dict["{}m{}".format(var_prefix,var)]=configs["category_str_to_num"][normal_val]
        elif normal_val in ["none","no"]:
            meta_normal_dict["{}m{}".format(var_prefix,var)]="NO"
        else:
            print("ERROR: Cannot convert normal value...")
            sys.exit(1)

        if int(var) in configs["categorical_vars_meta"]:
            meta_encoding_dict["{}m{}".format(var_prefix,var)]="categorical"
        elif int(var) in configs["ordinal_vars_meta"]:
            meta_encoding_dict["{}m{}".format(var_prefix,var)]="ordinal"
        elif int(var) in configs["binary_vars_meta"]:
            meta_encoding_dict["{}m{}".format(var_prefix,var)]="binary"
        else:
            meta_encoding_dict["{}m{}".format(var_prefix,var)]="continuous"

    for var in meta_lab_vars:
        df_red=df_labref[df_labref["MetaVariableID"]==int(var)]
        normal_val=df_red["NormalValue"].dropna().unique()

        assert(normal_val.size==1)
        normal_val=normal_val[0]
        if mlhc_math.is_numeric(normal_val):
            meta_normal_dict["vm{}".format(var)]=float(normal_val)
        else:
            print("ERROR: Could not determine the normal value...")
            sys.exit(1)

        if int(var) in configs["categorical_vars_meta"]:
            meta_encoding_dict["vm{}".format(var)]="categorical"
        elif int(var) in configs["ordinal_vars_meta"]:
            meta_encoding_dict["vm{}".format(var)]="ordinal"
        elif int(var) in configs["binary_vars_meta"]:
            meta_encoding_dict["vm{}".format(var)]="binary"
        else:
            meta_encoding_dict["vm{}".format(var)]="continuous"

    if not configs["debug_mode"]:
        mlhc_io.save_pickle(meta_normal_dict, configs["meta_normalval_map_path"])
        mlhc_io.save_pickle(meta_encoding_dict, configs["meta_varencoding_map_path"])

def parse_cmd_args():

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--current_varref_table", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/ref_excel/varref_excel_v6.tsv", 
                        help="Varref meta-data table")
    parser.add_argument("--current_labref_table", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/ref_excel/labref_excel_v6.tsv", 
                        help="Labref meta-data table")
    parser.add_argument("--var_list_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/var_list_v6.txt", 
                        help="List of non-lab variables")
    parser.add_argument("--labvar_list_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/labvar_list_v6.txt", 
                        help="List of lab variables")
    parser.add_argument("--meta_var_list_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/meta_var_list_v6.txt",
                        help="List of meta non-lab variables")
    parser.add_argument("--meta_labvar_list_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/meta_labvar_list_v6.txt",
                        help="List of meta lab variables")

    # Output paths
    parser.add_argument("--normalval_map_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/normalval_map_v6.pickle", 
                        help="Path where normal value dict is stored")
    parser.add_argument("--varencoding_map_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/varencoding_map_v6.pickle",
                        help="Path where the encoding dict is stored")
    parser.add_argument("--meta_normalval_map_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/meta_normalval_map_v6.pickle",
                        help="Path where normal value dict is stored for meta variables")
    parser.add_argument("--meta_varencoding_map_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/meta_varencoding_map_v6.pickle",
                        help="Path where the encoding dict is stored for meta variables")

    # Arguments
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Should script generate output on FS?")

    args=parser.parse_args()
    configs=vars(args)

    # DEFAULT ARGUMENTS

    # Special variables which are categorical
    configs["categorical_vars"]=[15001166, 3845, 15001552]

    # Special variables which are ordinal
    configs["ordinal_vars"]=[10000100, 10000200, 10000300, 15001565, 7887]

    # Special variables which are binary
    configs["binary_vars"]=[10002508]

    # Special variables that are categorical among the meta-variables
    configs["categorical_vars_meta"]=[19,60,66]

    # Special variables that are ordinal among the meta-variables
    configs["ordinal_vars_meta"]=[25,26,27,28,30,108]

    # Special variables that are binary using the meta-variables
    configs["binary_vars_meta"]=[38,47,52,53,54,55,56,57,
                                 67,68,70,71,72,73,74,75,
                                 76,79,85,87,88,90,92,93,
                                 94,96,97,98,99,100,101,102,
                                 103,104,105,106,107,109,
                                 110,111,112,113,114,115,116,117,
                                 118,119,120,121,122,123,124,
                                 125,126,127,128,129,130]

    # Mapping of string categories to numbers
    configs["category_str_to_num"]={"Sinus rhythm": 1.0}

    return configs

if __name__=="__main__":
    configs=parse_cmd_args()
    save_var_auxiliary_dicts(configs)
