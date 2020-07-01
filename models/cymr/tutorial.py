import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from psifr import fr
from cymr import models
from cymr import network

import synth_data_convenience as sdc

model_dir = '/Users/polyn/work/cfr'
#patterns_file = os.path.join(model_dir, 'cfr_patterns.hdf5')
#patterns = network.load_patterns(patterns_file)

# can we create a synthetic patterns dict, this would be informative
# a dict, with:
# 'items' and string array of item names
# 'vector' and sub-field 'loc' for localist vectors

# this function creates a pandas dataframe to use for generating
# synthetic data
patterns = sdc.create_patterns(24)

n_subj = 20
n_trials = 6
list_len = 24
synth_study = sdc.create_expt(patterns, n_subj, n_trials, list_len)

model = models.CMRDistributed()

# TODO: add documentation explaining what the parameters do
param = {'B_enc': 0.7, 'B_rec': 0.5, 'w_loc': 1, 'P1': 8, 'P2': 1, 'T': 0.35,
         'X1':0.001, 'X2': 0.5, 'Dfc': 3, 'Dcf': 1, 'Dff': 0,
         'Lfc': 1, 'Lcf': 1, 'Afc': 0, 'Acf': 0, 'Aff': 0, 'B_start': 0}
# TODO: add documentation of what this syntax means
weights = {'fcf': {'loc': 'w_loc'}}

# generate synthetic recall sequences
# simulates all the study trials created in the synth_study data frame
sim = model.generate(synth_study, param, patterns=patterns,weights=weights)

# merge the study and recall events
sim_merged = fr.merge_free_recall(sim)

# serial position curve
rec_pos = fr.spc(sim_merged)
g = fr.plot_spc(rec_pos)
plt.savefig('temp_spc.pdf')

# lag CRP curve
crp = fr.lag_crp(sim_merged)
g = fr.plot_lag_crp(crp)
g.set(ylim=(0, .6))
plt.savefig('temp_crp.pdf')


# Section 3. Using predictive simulations to perform parameter recovery

# parameter sweep B_rec, check cmr_cfr notebook

param_name = ['B_enc', 'B_rec']
param_sweep = [np.linspace(0, 1, 5), np.linspace(0, 1, 5)]

# this didn't work, added B_enc and it worked, added issue,
# seems like parameter_sweep function doesn't work for one param
#param_name = ['B_rec']
#param_sweep = [np.linspace(0, 1, 5)]

# uncomment this to develop
#results = model.parameter_sweep(synth_study, param, param_name, param_sweep,
#                                dependent=None, patterns=patterns, weights=weights, n_rep=1)
#sim = results.groupby(level=[0, 1]).apply(fr.merge_free_recall)

#p = sim.groupby(level=[0, 1]).apply(fr.spc)
#g = fr.plot_spc(p.reset_index(), row=param_names[0], col=param_names[1])

print('ho')


