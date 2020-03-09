''' 
Class wrapper for imputation, transforming an input 
Pandas data frame to an output Pandas data-frame
'''

import os
import sys
import os.path

import numpy as np
import scipy as sp
import pandas as pd

import circews.functions.forward_filling as bern_forward_fill
import circews.functions.util.array as mlhc_array
import circews.functions.util.math as mlhc_math

class TimegridderForwardFill:
    ''' 
    Function transforming the merged data representation of the ICU Bern data-set to 
    a time-gridded version. Special version that does not use the complex imputation schema but only forward
    fills indefinitely.
    '''
    def __init__(self, is_dim_reduced, grid_period=None, max_grid_length_days=None, use_adaptive_impute=True):
        self.is_dim_reduced=is_dim_reduced
        
        if grid_period is not None:
            self.grid_period=grid_period
        else:
            assert(False)

        if max_grid_length_days is not None:
            self.max_grid_length_days=max_grid_length_days
            self.max_grid_length_secs=self.max_grid_length_days*24*3600

        self.patient_id_key="PatientID"
        self.rel_datetime_key="RelDatetime"
        self.abs_datetime_key="AbsDatetime"

        # Default imputation parameters if the dictionary does not contain values
        self.default_fill_interval_secs=172800     # 2 DAYS (ARBITRARY)
        self.default_rolling_mean_secs=2*172800    # 4 DAYS (ARBITRARY)
        self.default_fill_interval_secs=172800     # 2 DAYS (ARBITRARY)

        if use_adaptive_impute is not None:
            self.use_adaptive_impute=use_adaptive_impute

        self.key_dict={"F": "female", "M": "male", "U": "unknown"}


    def set_static_table(self, df_static):
        ''' Set a static data table as context information in which we can lookup information 
            about a patient'''
        self.df_static=df_static


    def set_typical_weight_dict(self, typical_weight_dict):
        ''' Set a typical weight dictionary as context information to use during information'''
        self.typical_weight_dict=typical_weight_dict



    def set_median_bmi_dict(self, median_bmi_dict):
        ''' Set a median BMI dictionary as context information to use during imputation'''
        self.median_bmi_dict=median_bmi_dict



    def set_var_encoding_map(self, var_encoding_map):
        ''' Set a variable encoding map that specifies for each of the variables in the data-frame the variable encoding'''
        self.var_encoding_map=var_encoding_map


    def set_imputation_params(self, global_impute_dict, interval_median_dict, interval_iqr_dict):
        ''' Set global imputation parameters of the transformer'''
        self.global_impute_dict=global_impute_dict
        self.interval_median_dict=interval_median_dict
        self.interval_iqr_dict=interval_iqr_dict


    def set_normal_vals(self, normal_val_map):
        ''' Set clinically normal values to be used for global replacement in the imputation algorithm'''
        self.normal_dict=normal_val_map

        # Convert to numeric encoding for some special variables
        if self.is_dim_reduced:
            self.normal_dict["vm57"]=0.0 # ECMO
            self.normal_dict["vm60"]=0.0 # ServoiMode
            self.normal_dict["vm66"]=0.0 # AirwayCode
            self.normal_dict["vm72"]=0.0 # Haemofiltration
            self.normal_dict["vm131"]="CUSTOM_FORMULA" # Weight
        else:
            self.normal_dict["v3845"]=0.0 # ServoiMode
            self.normal_dict["v15001552"]=0.0 # AirwayCode
            self.normal_dict["v10002508"]=0.0 # Haemofiltration
            self.normal_dict["v10000400"]="CUSTOM_FORMULA" # Weight


    def _check_state(self):
        ''' Assert correct initialization before the transformer is used'''
        assert(self.typical_weight_dict is not None)
        assert(self.df_static is not None)
        assert(self.median_bmi_dict is not None)
        assert(self.var_encoding_map is not None)
        assert(self.global_impute_dict is not None)
        assert(self.interval_median_dict is not None)
        assert(self.interval_iqr_dict is not None)
        assert(self.normal_dict is not None)



    def transform(self, patient_df, pid=None):
        ''' Transformer method, taking as input a data-frame with irregularly sampled input data. The method 
            assumes that the data-frame contains a time-stamp column, and the data-frame is sorted along the first 
            axis in non-decreasing order with respect to the timestamp column. Pass the <pid> of the patient stay
            as additional information'''
        self._check_state()
        static_table=self.df_static[self.df_static["PatientID"]==pid]

        # No static data, patient is not valid, exclude on-the-fly
        if static_table.shape[0]==0:
            print("WARNING: No static data in patient table...")
            return None

        # More than one row, select one of the rows arbitrarily
        if static_table.shape[0]>1:
            print("WARNING: More than one row in static table...")
            static_table=static_table.take([0],axis=0)

        static_height=float(static_table["Height"])
        static_gender=str(static_table["Sex"].values[0]).strip()
        assert(static_gender in ["F","M","U"])

        if static_gender in ["F", "M"]:
            typical_weight=self.typical_weight_dict[static_gender]
        else:
            typical_weight=(self.typical_weight_dict["M"]+self.typical_weight_dict["F"])/2.0

        personal_bmi=self.median_bmi_dict[self.key_dict[static_gender]]

        ## If either the endpoints or the features don't exist, log the failure but do nothing, the missing patients can be
        #  latter added as a new group to the output H5
        if patient_df.shape[0]==0:
            print("WARNING: p{} has missing features, skipping output generation...".format(pid))
            return None

        all_keys=list(set(patient_df.columns.values.tolist()).difference(set(["Datetime","PatientID","a_temp","m_pm_1", "m_pm_2"])))
        
        ts=patient_df["Datetime"]
        ts_arr=np.array(ts)
        n_ts=ts_arr.size

        if self.is_dim_reduced:
            hr=np.array(patient_df["vm1"])
        else:
            hr=np.array(patient_df["v200"])

        finite_hr=ts_arr[np.isfinite(hr)]

        if finite_hr.size==0:
            print("WARNING: Patient {} has no HR, ignoring patient...".format(pid))
            return None

        ts_min=ts_arr[np.isfinite(hr)][0]
        ts_max=ts_arr[np.isfinite(hr)][-1]
        max_ts_diff=(ts_max-ts_min)/np.timedelta64(1,'s')

        time_grid=np.arange(0.0,min(max_ts_diff+1.0,self.max_grid_length_secs),self.grid_period)
        time_grid_abs=[ts_min+pdts.Timedelta(seconds=time_grid[idx]) for idx in range(time_grid.size)]
        imputed_df_dict={}
        imputed_df_dict[self.patient_id_key]=[int(pid)]*time_grid.size
        imputed_df_dict[self.rel_datetime_key]=time_grid
        imputed_df_dict[self.abs_datetime_key]=time_grid_abs

        ## There is nothing to do if the patient has no records, just return...
        if n_ts==0:
            print("WARNING: p{} has an empty record, skipping output generation...".format(patient))
            return None

        ## Initialize the storage for the imputed time grid, NANs for the non-pharma, 0 for pharma.
        for col in all_keys:
            if col[0]=="p":
                imputed_df_dict[col]=np.zeros(time_grid.size)
            elif col[0]=="v":
                imputed_df_dict[col]=mlhc_array.empty_nan(time_grid.size)
            else:
                print("ERROR: Invalid variable type")
                assert(False)

        imputed_df=pd.DataFrame(imputed_df_dict)
        norm_ts=np.array(ts-ts_min)/np.timedelta64(1,'s')

        # Schedule for order of variable imputation
        if self.is_dim_reduced:
            all_keys.remove("vm131")
            all_keys=["vm131"]+all_keys
        else:
            all_keys.remove("v10000400")
            all_keys=["v10000400"]+all_keys

        ## Impute all variables independently, with the two relevant cases pharma variable and other variable,
        #  distinguishable from the variable prefix. We enforce that weight is the first variable to be imputed, so that 
        #  its time-gridded information can later be used by other custom formulae imputations that depend on it.
        for var_idx,variable in enumerate(all_keys):
            df_var=patient_df[variable]
            assert(n_ts==df_var.shape[0]==norm_ts.size)

            ## Non-pharma variable case
            if variable[0]=="v":
                valid_normal=False
                var_encoding=self.var_encoding_map[variable]

                # Saved a value in the dict of normal values
                if variable in self.normal_dict:
                    saved_normal_var=self.normal_dict[variable]

                    # Saved normal value is already numeric, no need to encode it here...
                    if mlhc_math.is_numeric(saved_normal_var) and np.isfinite(saved_normal_var):
                        global_impute_val=saved_normal_var
                        valid_normal=True

                # Could not determine a valid normal value, have to fall back to pre-computed global statistic
                if not valid_normal:

                    # Fill in the weight using BMI calculations
                    if variable in ["v10000400","vm131"]:

                        # If we have an observed height can use BMI
                        if np.isfinite(static_height):
                            global_impute_val=personal_bmi*(static_height/100)**2
                        else:
                            global_impute_val=typical_weight

                    # Fill in with the global statistic
                    elif variable in self.global_impute_dict:
                        global_impute_val=self.global_impute_dict[variable]

                    # Rare case, no observation in the imputation data-set
                    else:
                        global_impute_val=np.nan

                # Default values where median/IQR interval not saved
                if variable not in self.interval_median_dict:
                    fill_interval_secs=self.default_fill_interval_secs
                    rolling_mean_secs=self.default_rolling_mean_secs
                    fill_interval_secs=self.default_fill_interval_secs

                # We have to impose minimum period to have boundary conditions where the backward window for 
                # slope estimation is empty or an observation is not even filled to the next grid point to the right.
                else:
                    med_interval=self.interval_median_dict[variable]
                    iqr_interval=self.interval_iqr_dict[variable]
                    base_val=med_interval+2*iqr_interval
                    fill_interval_secs=max(self.grid_period, base_val)
                    rolling_mean_secs=max(2*self.grid_period, 2*base_val)
                    return_mean_secs=max(2*self.grid_period, base_val)

                raw_col=np.array(df_var)
                assert(raw_col.size==norm_ts.size)
                observ_idx=np.isfinite(raw_col)
                observ_ts=norm_ts[observ_idx]
                observ_val=raw_col[observ_idx]

                ## No values have been observed for this variable, it has to be imputed using the global mean
                if observ_val.size==0:
                    est_vals=mlhc_array.value_empty(time_grid.size,global_impute_val)
                    imputed_df[variable]=est_vals
                    imputed_df["{}_IMPUTED_STATUS_CUM_COUNT".format(variable)]=np.zeros(time_grid.size)
                    imputed_df["{}_IMPUTED_STATUS_TIME_TO".format(variable)]=mlhc_array.value_empty(time_grid.size,-1.0)
                    continue

                assert(np.isfinite(observ_val).all())
                assert(np.isfinite(observ_ts).all())

                if self.use_adaptive_impute:

                    # Formulae imputation
                    if variable in ["v1000","v1010","v10020000","v30005010","v30005110","vm13","vm24","vm31","vm32"]:
                        existing_weight_col=np.array(imputed_df["vm131"]) if self.is_dim_reduced else np.array(imputed_df["v10000400"])
                        est_vals,cum_count_ts,time_to_last_ms=bern_forward_fill.impute_forward_fill_new_only_ffill(observ_ts,observ_val,time_grid, global_impute_val, self.grid_period,
                                                                                                                   fill_interval_secs= fill_interval_secs, rolling_mean_secs= rolling_mean_secs, 
                                                                                                                   return_mean_secs= return_mean_secs, var_type="non_pharma", var_encoding=var_encoding,
                                                                                                                   variable_id=variable, weight_imputed_col=existing_weight_col, static_height=static_height,
                                                                                                                   personal_bmi=personal_bmi)
                    elif variable in ["v10000400","vm131"]:
                        est_vals,cum_count_ts,time_to_last_ms=bern_forward_fill.impute_forward_fill_new_only_ffill(observ_ts, observ_val, time_grid, global_impute_val, self.grid_period,var_type="weight")
                    else:
                        est_vals,cum_count_ts,time_to_last_ms=bern_forward_fill.impute_forward_fill_new_only_ffill(observ_ts,observ_val,time_grid, global_impute_val, self.grid_period,
                                                                                                                   fill_interval_secs= fill_interval_secs, rolling_mean_secs= rolling_mean_secs, 
                                                                                                                   return_mean_secs= return_mean_secs, var_type="non_pharma", var_encoding=var_encoding,
                                                                                                                   variable_id=variable)

                else:
                    assert(False)
                    est_vals=bern_forward_fill.impute_forward_fill(observ_ts, observ_val, time_grid, global_mean_var)

                assert(np.isnan(global_impute_val) or np.isfinite(est_vals).all())
                imputed_df[variable]=est_vals
                imputed_df["{}_IMPUTED_STATUS_CUM_COUNT".format(variable)]=cum_count_ts
                imputed_df["{}_IMPUTED_STATUS_TIME_TO".format(variable)]=time_to_last_ms


            ## Pharma variable case, the doses have to be recomputed to the time-grid. The global imputation value is 0, because the rate assumed w/o observation
            #  is 0 (no medication flow)
            elif variable[0]=="p":
                global_impute_val=0.0
                raw_col=np.array(df_var)
                assert(raw_col.size==norm_ts.size)
                observ_idx=np.isfinite(raw_col)
                observ_ts=norm_ts[observ_idx]
                observ_val=raw_col[observ_idx]

                ## No values have been observed for this pharma-variable, leave Zero in this series
                if observ_val.size==0:
                    continue

                assert(np.isfinite(observ_val).all())
                assert(np.isfinite(observ_ts).all())
                est_vals,cum_count_ts,time_to_last_ms=bern_forward_fill.impute_forward_fill_new_only_ffill(observ_ts, observ_val, time_grid, global_impute_val, self.grid_period,var_type="pharma")
                assert(np.isfinite(est_vals).all())
                imputed_df[variable]=est_vals
            else:
                print("ERROR: Invalid variable, exiting...")
                assert(False)

        return imputed_df


    
        
        
