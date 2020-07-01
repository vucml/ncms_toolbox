
import numpy as np
import pandas as pd
from string import ascii_letters

def create_patterns(pool_size):
    # goal here is to have a bare-bones set of patterns that makes
    # it clear what fields cymr expects and what format they should have
    # pool_size must be an int
    # create a pool of fake item strings using ascii_letters
    # which means currently pool_size must be < 52
    item_pool = np.empty((pool_size,), dtype=str)
    for i in range(pool_size):
        item_pool[i] = ascii_letters[i]
    # create a set of localist vectors:
    localist_dict = {'loc': np.eye(pool_size)}
    patterns = {'items': item_pool,
                'vector': localist_dict}
    return patterns

def create_expt(patterns, n_subj, n_trials, list_len):
    for i in range(n_subj):
        sess_frame = create_session(patterns, i+1, n_trials, list_len)
        if i==0:
            expt_frame = sess_frame.copy()
        else:
            expt_frame = expt_frame.append(sess_frame)
    return expt_frame

def create_session(patterns, subjid,n_trials,list_len):
    for i in range(n_trials):
        trial_frame = create_list(patterns, subjid, i+1, list_len)
        if i==0:
            sess_frame = trial_frame.copy()
        else:
            sess_frame = sess_frame.append(trial_frame)
    return sess_frame


def create_list(patterns, subjid,trialnum,list_len):

    positions = np.zeros((list_len,),dtype=int)
    indices = np.zeros((list_len,), dtype=int)
    item_names = np.empty((list_len,), dtype=str)
    for i in range(list_len):
        positions[i] = i+1
        indices[i] = i
        item_names[i] = patterns['items'][i]

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
