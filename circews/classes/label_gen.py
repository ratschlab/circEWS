''' 
Class wrapper for label generation, transforming an input data-frame with endpoints and
an input data-frame with imputed data to a Pandas data-frame
with labels
'''

import os
import sys
import os.path
import ipdb

import numpy as np
import scipy as sp
import pandas as pd

import circews.functions.labels as bern_labels

class AllLabel:
    '''
    Annotate all labels jointly, including full WorseState and WorseStateSoft labels, multi-class
    classification labels, regression-time-to-event labels, and smaller component labels
    that refer to conditions on MAP, Lactate and the medications.
    '''

    def __init__(self, lhours, rhours, dataset=None):
        self.abs_datetime_key="AbsDatetime"
        self.rel_datetime_key="RelDatetime"
        self.patient_id_key="PatientID"
        self.lhours=lhours
        self.rhours=rhours
        self.label_key="AllLabels{}To{}Hours".format(self.lhours, self.rhours)
        self.grid_step_seconds=300.0
        self.dataset=dataset

    def transform(self, df_pat, df_endpoint, pid=None):
        abs_time_col=df_pat[self.abs_datetime_key]
        rel_time_col=df_pat[self.rel_datetime_key]
        patient_col=df_pat[self.patient_id_key]
        
        if df_pat.shape[0]==0 or df_endpoint.shape[0]==0:
            print("WARNING: Patient {} has no impute data, skipping...".format(pid), flush=True)
            return None

        df_endpoint.set_index(keys="Datetime", inplace=True, verify_integrity=True)
        
        try:
            if self.dataset=="bern":
                df_endpoint=df_endpoint.reindex(index=df_pat.AbsDatetime,method="nearest")
            elif self.dataset=="mimic":
                df_endpoint=df_endpoint.reindex(index=df_pat.AbsDatetime,method="ffill")

        except:
            print("WARNING: Issue when re-indexing frame of patient: {}".format(pid), flush=True)
            return None

        endpoint_status_arr=np.array(df_endpoint.endpoint_status)
        unique_status=np.unique(endpoint_status_arr)

        for status in unique_status:
            assert(status in ["unknown","event 0","event 1", "event 2", "event 3",
                              "maybe 1","maybe 2", "maybe 3","probably not 1", "probably not 2", "probably not 3"])

        lactate_above_ts=np.array(df_endpoint.lactate_above_threshold,dtype=np.float)
        map_below_ts=np.array(df_endpoint.MAP_below_threshold,dtype=np.float)
        l1_present=np.array(df_endpoint.level1_drugs_present,dtype=np.float)
        l2_present=np.array(df_endpoint.level2_drugs_present,dtype=np.float)
        l3_present=np.array(df_endpoint.level3_drugs_present,dtype=np.float)

        worse_state_arr=bern_labels.future_worse_state(endpoint_status_arr, self.lhours, self.rhours, self.grid_step_seconds) # Joint (A|D|E)
        worse_state_soft_arr=bern_labels.future_worse_state_soft(endpoint_status_arr, self.lhours, self.rhours, self.grid_step_seconds) # Joint (B|C|D|E)
        from_0_arr=bern_labels.future_worse_state_from_0(endpoint_status_arr, self.lhours, self.rhours, self.grid_step_seconds) # Separate A
        from_0_soft_arr=bern_labels.future_worse_state_soft_from_0(endpoint_status_arr, self.lhours, self.rhours, self.grid_step_seconds) # Separate B
        from_probably_not_arr=bern_labels.future_worse_state_from_pn(endpoint_status_arr, self.lhours, self.rhours, self.grid_step_seconds) # Separate C
        from_1_arr=bern_labels.future_worse_state_from_1(endpoint_status_arr, self.lhours, self.rhours, self.grid_step_seconds) # Separate D
        from_2_arr=bern_labels.future_worse_state_from_2(endpoint_status_arr, self.lhours, self.rhours, self.grid_step_seconds) # Separate E;
        from_1_or_2_arr=bern_labels.future_worse_state_from_1_or_2(endpoint_status_arr, self.lhours, self.rhours, self.grid_step_seconds) # Join(D|E)

        lactate_any_arr=bern_labels.any_positive_transition(lactate_above_ts, self.lhours, self.rhours, self.grid_step_seconds)
        map_any_arr=bern_labels.any_positive_transition(map_below_ts, self.lhours, self.rhours, self.grid_step_seconds)
        l1_drugs_any_arr=bern_labels.any_positive_transition(l1_present, self.lhours, self.rhours, self.grid_step_seconds)
        l2_drugs_any_arr=bern_labels.any_positive_transition(l2_present, self.lhours, self.rhours, self.grid_step_seconds)
        l3_drugs_any_arr=bern_labels.any_positive_transition(l3_present, self.lhours, self.rhours, self.grid_step_seconds)
        time_to_worse_state_binned_arr=bern_labels.time_to_worse_state_binned(endpoint_status_arr, self.lhours, self.rhours, self.grid_step_seconds)
        time_to_worse_state_arr=bern_labels.time_to_worse_state(endpoint_status_arr, self.lhours, self.rhours, self.grid_step_seconds)

        output_df_dict={}
        output_df_dict[self.abs_datetime_key]=abs_time_col
        output_df_dict[self.rel_datetime_key]=rel_time_col
        output_df_dict[self.patient_id_key]=patient_col

        output_df_dict["WorseState{}To{}Hours".format(self.lhours, self.rhours)]=worse_state_arr
        output_df_dict["WorseStateSoft{}To{}Hours".format(self.lhours, self.rhours)]=worse_state_soft_arr
        output_df_dict["WorseStateFromZero{}To{}Hours".format(self.lhours, self.rhours)]=from_0_arr
        output_df_dict["WorseStateSoftFromZero{}To{}Hours".format(self.lhours, self.rhours)]=from_0_soft_arr
        output_df_dict["WorseStateFromPn{}To{}Hours".format(self.lhours, self.rhours)]=from_probably_not_arr
        output_df_dict["WorseStateFromOne{}To{}Hours".format(self.lhours, self.rhours)]=from_1_arr
        output_df_dict["WorseStateFromTwo{}To{}Hours".format(self.lhours, self.rhours)]=from_2_arr
        output_df_dict["WorseStateFromOneOrTwo{}To{}Hours".format(self.lhours, self.rhours)]=from_1_or_2_arr

        output_df_dict["LactateAboveTs{}To{}Hours".format(self.lhours, self.rhours)]=lactate_any_arr
        output_df_dict["MAPBelowTs{}To{}Hours".format(self.lhours, self.rhours)]=map_any_arr
        output_df_dict["L1Drugs{}To{}Hours".format(self.lhours, self.rhours)]=l1_drugs_any_arr
        output_df_dict["L2Drugs{}To{}Hours".format(self.lhours, self.rhours)]=l2_drugs_any_arr
        output_df_dict["L3Drugs{}To{}Hours".format(self.lhours, self.rhours)]=l3_drugs_any_arr
        output_df_dict["TimeToWorseState{}To{}Hours".format(self.lhours, self.rhours)]=time_to_worse_state_arr
        output_df_dict["TimeToWorseStateBinned{}To{}Hours".format(self.lhours, self.rhours)]=time_to_worse_state_binned_arr

        output_df=pd.DataFrame(output_df_dict)
        return output_df


class DeteriorationLabel:

    def __init__(self,lhours,rhours):
        self.abs_datetime_key="AbsDatetime"
        self.rel_datetime_key="RelDatetime"
        self.patient_id_key="PatientID"
        self.lhours=lhours
        self.rhours=rhours
        self.label_key="Deterioration_{}To{}Hours".format(self.lhours,self.rhours)
        self.grid_step_seconds=300.0

    
    def transform(self, df_pat, df_endpoint, pid=None):
        abs_time_col=df_pat[self.abs_datetime_key]
        rel_time_col=df_pat[self.rel_datetime_key]
        patient_col=df_pat[self.patient_id_key]

        ## Patient has no information in the imputed table or the endpoints (SHOULD NOT HAPPEN)
        if df_pat.shape[0]==0 or df_endpoint.shape[0]==0:
            print("WARNING: Patient {} has no impute data, skipping...".format(pid),flush=True)
            return None

        df_endpoint.set_index(keys="Datetime",inplace=True,verify_integrity=True)

        # Re-index the endpoint to the grid of the imputed data.
        try:
            df_endpoint=df_endpoint.reindex(index=df_pat.AbsDatetime,method="nearest")
        except :
            print("WARNING: Issue when re-indexing frame of patient {}".format(pid),flush=True)
            return None

        event1_arr=np.array(df_endpoint.event1)
        event2_arr=np.array(df_endpoint.event2)
        event3_arr=np.array(df_endpoint.event3)
        maybe1_arr=np.array(df_endpoint.maybe_event1)
        maybe2_arr=np.array(df_endpoint.maybe_event2)
        maybe3_arr=np.array(df_endpoint.maybe_event3)
        not1_arr=np.array(df_endpoint.probably_not_event1)
        not2_arr=np.array(df_endpoint.probably_not_event2)
        not3_arr=np.array(df_endpoint.probably_not_event3)

        # Any deterioration, does not require that the exact downward takes place in the forward horizon, but only if there 
        # is some more severe endpoint in the period

        if self.lhours==0:
            label_arr=bern_labels.future_worse_state(event1_arr, event2_arr, event3_arr,maybe1_arr, maybe2_arr, maybe3_arr, self.lhours, self.rhours, self.grid_step_seconds)
        else:
            label_arr=bern_labels.future_deterioration(event1_arr, event2_arr, event3_arr,maybe1_arr, maybe2_arr, maybe3_arr, self.lhours, self.rhours, self.grid_step_seconds)
        
        output_df_dict={}
        output_df_dict[self.abs_datetime_key]=abs_time_col
        output_df_dict[self.rel_datetime_key]=rel_time_col
        output_df_dict[self.patient_id_key]=patient_col
        output_df_dict[self.label_key]=label_arr
        output_df=pd.DataFrame(output_df_dict)

        return output_df


class WorseStateLabel:
    
    def __init__(self, lhours, rhours):
        self.abs_datetime_key="AbsDatetime"
        self.rel_datetime_key="RelDatetime"
        self.patient_id_key="PatientID"
        self.lhours=lhours
        self.rhours=rhours
        self.label_key="WorseState_{}To{}Hours".format(float(self.lhours),float(self.rhours))
        self.grid_step_seconds=300.0


    def transform(self, df_pat, df_endpoint,pid=None):
        abs_time_col=df_pat[self.abs_datetime_key]
        rel_time_col=df_pat[self.rel_datetime_key]
        patient_col=df_pat[self.patient_id_key]

        ## Patient has no information in the imputed table or the endpoints (SHOULD NOT HAPPEN)
        if df_pat.shape[0]==0 or df_endpoint.shape[0]==0:
            print("WARNING: Patient {} has no impute data, skipping...".format(pid),flush=True)
            return None

        df_endpoint.set_index(keys="Datetime",inplace=True,verify_integrity=True)

        # Re-index the endpoint to the grid of the imputed data.
        try:
            df_endpoint=df_endpoint.reindex(index=df_pat.AbsDatetime,method="nearest")
        except :
            print("WARNING: Issue when re-indexing frame of patient {}".format(pid),flush=True)
            return None

        event1_arr=np.array(df_endpoint.event1)
        event2_arr=np.array(df_endpoint.event2)
        event3_arr=np.array(df_endpoint.event3)
        maybe1_arr=np.array(df_endpoint.maybe_event1)
        maybe2_arr=np.array(df_endpoint.maybe_event2)
        maybe3_arr=np.array(df_endpoint.maybe_event3)
        not1_arr=np.array(df_endpoint.probably_not_event1)
        not2_arr=np.array(df_endpoint.probably_not_event2)
        not3_arr=np.array(df_endpoint.probably_not_event3)

        # Any deterioration, does not require that the exact downward takes place in the forward horizon, but only if there 
        # is some more severe endpoint in the period
        label_arr=bern_labels.future_worse_state(event1_arr, event2_arr, event3_arr, maybe1_arr, maybe2_arr, maybe3_arr, not1_arr, not2_arr,
                                                 not3_arr, self.lhours, self.rhours, self.grid_step_seconds)
        
        output_df_dict={}
        output_df_dict[self.abs_datetime_key]=abs_time_col
        output_df_dict[self.rel_datetime_key]=rel_time_col
        output_df_dict[self.patient_id_key]=patient_col
        output_df_dict[self.label_key]=label_arr
        output_df=pd.DataFrame(output_df_dict)

        return output_df

class WorseStateSoftLabel:
    
    def __init__(self, lhours, rhours):
        self.abs_datetime_key="AbsDatetime"
        self.rel_datetime_key="RelDatetime"
        self.patient_id_key="PatientID"
        self.lhours=lhours
        self.rhours=rhours
        self.label_key="WorseState_{}To{}Hours".format(float(self.lhours),float(self.rhours))
        self.grid_step_seconds=300.0


    def transform(self, df_pat, df_endpoint,pid=None):
        abs_time_col=df_pat[self.abs_datetime_key]
        rel_time_col=df_pat[self.rel_datetime_key]
        patient_col=df_pat[self.patient_id_key]

        ## Patient has no information in the imputed table or the endpoints (SHOULD NOT HAPPEN)
        if df_pat.shape[0]==0 or df_endpoint.shape[0]==0:
            print("WARNING: Patient {} has no impute data, skipping...".format(pid),flush=True)
            return None

        df_endpoint.set_index(keys="Datetime",inplace=True,verify_integrity=True)

        # Re-index the endpoint to the grid of the imputed data.
        try:
            df_endpoint=df_endpoint.reindex(index=df_pat.AbsDatetime,method="nearest")
        except :
            print("WARNING: Issue when re-indexing frame of patient {}".format(pid),flush=True)
            return None

        event1_arr=np.array(df_endpoint.event1)
        event2_arr=np.array(df_endpoint.event2)
        event3_arr=np.array(df_endpoint.event3)
        maybe1_arr=np.array(df_endpoint.maybe_event1)
        maybe2_arr=np.array(df_endpoint.maybe_event2)
        maybe3_arr=np.array(df_endpoint.maybe_event3)
        not1_arr=np.array(df_endpoint.probably_not_event1)
        not2_arr=np.array(df_endpoint.probably_not_event2)
        not3_arr=np.array(df_endpoint.probably_not_event3)

        # Any deterioration, does not require that the exact downward takes place in the forward horizon, but only if there 
        # is some more severe endpoint in the period
        label_arr=bern_labels.future_worse_state_soft(event1_arr, event2_arr, event3_arr, maybe1_arr, maybe2_arr, maybe3_arr, not1_arr, not2_arr,
                                                      not3_arr, self.lhours, self.rhours, self.grid_step_seconds)
        
        output_df_dict={}
        output_df_dict[self.abs_datetime_key]=abs_time_col
        output_df_dict[self.rel_datetime_key]=rel_time_col
        output_df_dict[self.patient_id_key]=patient_col
        output_df_dict[self.label_key]=label_arr
        output_df=pd.DataFrame(output_df_dict)

        return output_df
