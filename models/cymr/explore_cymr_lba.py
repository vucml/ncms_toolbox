
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from psifr import fr
from cymr import models
from cymr import network
from cymr import cmr_lba

import synth_data_convenience as sdc

patterns = sdc.create_patterns(20)
n_subj = 20
n_trials = 6
list_len = 20
synth_study = sdc.create_expt(patterns, n_subj, n_trials, list_len)

# create the model
model = cmr_lba.CMRLBA()
param = {'B_enc': 0.7, 'B_rec': 0.5, 'w_loc': 1, 'P1': 8, 'P2': 1, 'T': 0.35,
         'X1':0.001, 'X2': 0.5, 'Dfc': 3, 'Dcf': 1, 'Dff': 0,
         'Lfc': 1, 'Lcf': 1, 'Afc': 0, 'Acf': 0, 'Aff': 0, 'B_start': 0}
weights = {'fcf': {'loc': 'w_loc'}}


A = 5
b = 10
s = 1
tau = 0
# v is vector of support for each item
v = [1, 2, 5]

resp = np.zeros((1000,), dtype=int)
rt = np.zeros((1000,), dtype=float)

for i in range(1000):
    resp[i], rt[i] = network.sample_response_lba(A, b, v, s, tau)

# fig, ax = plt.subplots()
# ax.hist(rt,50)
# fig.show()
# print(f'Response ID: {resp} Response time: {rt}\n')

recall_time_limit = 40.
# A: upper end of start point distribution
param['A'] = 5
# b: response threshold
param['b'] = 10
# s: st dev of drift rate samples
param['s'] = 1
# tau: non-decision time
param['tau'] = 0
param['recall_time_limit'] = recall_time_limit
sim_data = model.generate(synth_study, param, patterns=patterns, weights=weights)

# what is the likelihood of the synthetic data with the true generating model
logl, n = model.likelihood(sim_data, param, patterns=patterns, weights=weights)

# does it behave, does logl drop if params are altered (yes)
alt_param = param.copy()
alt_param['B_enc'] = 0.5
alt_logl, n = model.likelihood(sim_data, alt_param, patterns=patterns, weights=weights)
print(f'Orig param: {logl}\n Alt param: {alt_logl}\n')

# merge the study and recall events in preparation for analysis
sim_merged = fr.merge_free_recall(sim_data,recall_keys=['rt'])

# serial position curve
rec_pos = fr.spc(sim_merged)
g = fr.plot_spc(rec_pos)
plt.show()

# removes repeats
clean = sim_merged.query('study')
# just recall events
clean_rec = clean.query('recall')

# this tells you how many observations per output position
# clean_rec['output'].value_counts()

# subselection of 2 columns, grouping by the output position
# gives mean rt by out pos, which rises linearly
out_rt = clean_rec[['output','rt']].groupby('output').mean()
# resp times rise linearly with output position
plt.plot(out_rt)
plt.show()



print('hi')