'''
Class wrapper for imputation of static data
'''

import os 
import os.path
import sys
import ipdb

import numpy as np

class StaticDataImputer:

    def __init__(self, dataset=None):
        self.dataset=dataset

    def transform(self, df_static,df_train=None):
        ''' Imputes static data and returns imputed table with no NAN entries, does not encode the data
            yet but only removes NANs'''
        age_median=int(df_train["Age"].median())
        apache_mode=int(df_train["APACHECode"].mode())
        discharge_mode=int(df_train["Discharge"].mode())
        emergency_mode=int(df_train["Emergency"].mode())
        euroscores_mode=int(df_train["Euroscores"].mode())
        surgical_mode=int(df_train["Surgical"].mode())
        height_median=float(df_train["Height"].median())
        pat_group_mode=int(df_train["PatGroup"].mode())
        apache_pat_group_mode=int(df_train["APACHEPatGroup"].mode())
        sex_mode=df_train["Sex"].mode()

        if self.dataset=="bern":
            fill_dict={"APACHECode": apache_mode, "Discharge": discharge_mode, 
                       "Emergency": emergency_mode, "Euroscores": euroscores_mode, 
                       "Surgical": surgical_mode, "Height": height_median,
                       "Age": age_median, "PatGroup": pat_group_mode, "Sex": sex_mode,
                       "APACHEPatGroup": apache_pat_group_mode}
            df_static_imputed=df_static.fillna(value=fill_dict)
            return df_static_imputed[["PatientID", "Age", "APACHECode", "Discharge", "Emergency", "Euroscores", "Surgical", "Height","PatGroup","APACHEPatGroup","Sex"]]

        elif self.dataset=="mimic":
            fill_dict={"Emergency": emergency_mode, 
                       "Surgical": surgical_mode, "Height": height_median,
                       "Age": age_median, "Sex": sex_mode}
            df_static_imputed=df_static.fillna(value=fill_dict)
            return df_static_imputed[["PatientID", "Age", "Emergency", "Surgical", "Height","Sex"]]

        else:
            print("ERROR: Wrong data-set was specified")
            sys.exit(1)



