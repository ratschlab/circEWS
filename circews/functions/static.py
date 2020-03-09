
import numpy as np

def lookup_admission_time(pid, df_patient_full):
    ''' Looks up a proxy to admission time for a PID'''
    df_patient=df_patient_full[df_patient_full["PatientID"]==pid]
    if not df_patient.shape[0]==1:
        return None
    adm_time=np.array(df_patient["AdmissionTime"])[0]
    return adm_time
