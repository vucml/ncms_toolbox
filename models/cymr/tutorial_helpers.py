
import numpy as np
import pandas as pd
from psifr import fr
import seaborn as sns
import matplotlib.pyplot as plt
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

def create_expt(patterns, n_subj, n_trials, list_len, dummy_recalls=False):
    for i in range(n_subj):
        sess_frame = create_session(patterns, i+1, n_trials, list_len, dummy_recalls)
        if i==0:
            expt_frame = sess_frame.copy()
        else:
            expt_frame = expt_frame.append(sess_frame)
    return expt_frame

def create_session(patterns, subjid, n_trials, list_len, dummy_recalls):
    for i in range(n_trials):
        trial_frame = create_list(patterns, subjid, i+1, list_len, dummy_recalls)
        if i==0:
            sess_frame = trial_frame.copy()
        else:
            sess_frame = sess_frame.append(trial_frame)
    return sess_frame


def create_list(patterns, subjid, trialnum, list_len, dummy_recalls):

    positions = np.zeros((list_len,),dtype=int)
    indices = np.zeros((list_len,), dtype=int)
    item_names = np.empty((list_len,), dtype=str)
    for i in range(list_len):
        positions[i] = i+1
        indices[i] = i
        item_names[i] = patterns['items'][i]

    study_frame = pd.DataFrame({
        'subject': np.ones((list_len,),dtype=int) * subjid,
        'list': np.ones((list_len,),dtype=int) * trialnum,
        'trial_type': ['study'] * list_len,
        'position': positions,
        'item_index': indices,
        'item': item_names
    })

    if dummy_recalls:
        recall_frame = pd.DataFrame({
            'subject': np.ones((list_len,),dtype=int) * subjid,
            'list': np.ones((list_len,),dtype=int) * trialnum,
            'trial_type': ['recall'] * list_len,
            'position': positions,
            'item_index': indices,
            'item': item_names
        })
        trial_frame = study_frame.append(recall_frame)
    else:
        trial_frame = study_frame

    return trial_frame

def fix_hcmp_field(orig_df, new_df):
    # to identify a recall event, subject, list, trial_type, position
    for index, row in orig_df.iterrows():
        if row.trial_type == 'recall':
            # find the corresponding row or rows in dyn_sim (rows if n_rep > 1)
            mask = (new_df.subject==row.subject) & \
                   (new_df.list==row.list) & \
                   (new_df.position==row.position) & \
                   (new_df.trial_type=='recall')
            new_df.loc[mask, 'hcmp'] = row.hcmp
    return new_df

def plot_var_crp(df, figpath):
    crp1 = fr.lag_crp(df, test_key='hcmp', test=lambda x, y: x < -0.5)
    crp1['condition'] = 'low_brec'
    crp2 = fr.lag_crp(df, test_key='hcmp', test=lambda x, y: x > 0.5)
    crp2['condition'] = 'high_brec'
    combined = pd.concat([crp1, crp2])
    sns.set_theme(font_scale=1.2, style="ticks")
    max_lag = 5
    filt_neg = f'{-max_lag} <= lag < 0'
    filt_pos = f'0 < lag <= {max_lag}'
    g = sns.FacetGrid(combined, height=5)
    g.map_dataframe(
        lambda data, **kws: sns.lineplot(
            data=data.query(filt_neg), x='lag', y='prob', hue='condition', **kws)
    )
    g.map_dataframe(
        lambda data, **kws: sns.lineplot(
            data=data.query(filt_pos), x='lag', y='prob', hue='condition', **kws)
    )
    g.set_xlabels('Lag')
    g.set_ylabels('Cond. Resp. Prob.')
    plt.legend(['Low', 'High'], title='Temp. Reinst.')
    g.set(ylim=(0, 0.6))
    plt.savefig(figpath+'var_hcmp_crp.pdf', bbox_inches='tight')

def calc_aic(n, V, L):
    aic = -2*L + 2*V + (2*V*(V+1)) / (n-V-1)
    return aic
