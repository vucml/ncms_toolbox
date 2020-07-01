
import numpy as np
import pandas as pd
from string import ascii_lowercase

def create_expt(n_subj, n_trials, list_len):
    for i in range(n_subj):
        sess_frame = create_session(i+1, n_trials, list_len)
        if i==0:
            expt_frame = sess_frame.copy()
        else:
            expt_frame = expt_frame.append(sess_frame)
    return expt_frame

def create_session(subjid,n_trials,list_len):
    for i in range(n_trials):
        trial_frame = create_list(subjid, i+1, list_len)
        if i==0:
            sess_frame = trial_frame.copy()
        else:
            sess_frame = sess_frame.append(trial_frame)
    return sess_frame


def create_list(subjid,trialnum,list_len):

    positions = np.zeros((list_len,),dtype=int)
    indices = np.zeros((list_len,), dtype=int)
    item_names = np.empty((list_len,), dtype=str)
    for i in range(list_len):
        positions[i] = i+1
        indices[i] = i+1
        item_names[i] = ascii_lowercase[i]

    trial_frame = pd.DataFrame({
        'subject': np.ones((list_len,),dtype=int) * subjid,
        'list': np.ones((list_len,),dtype=int) * trialnum,
        'trial_type': ['study'] * list_len,
        'position': positions,
        'item_index': indices,
        'item': item_names
    })

    return trial_frame

#this = create_expt(2,3,5)
#this
