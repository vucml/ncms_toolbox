import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from psifr import fr
from cymr import models
from cymr import network

import synth_data_convenience as sdc

# Section 1.  Setting model parameters and task details

model_dir = '/Users/polyn/work/cfr'

# first we create a set of synthetic patterns corresponding to the
# set of items that will be on a study list
patterns = sdc.create_patterns(24)

# create a pandas dataframe with a set of synthetic study events,
# these will be used to generate synthetic recall sequences
n_subj = 20
n_trials = 6
list_len = 24
synth_study = sdc.create_expt(patterns, n_subj, n_trials, list_len)

# create the model
model = models.CMRDistributed()

# define the parameters
# TODO: add documentation explaining what the parameters do
param = {'B_enc': 0.7, 'B_rec': 0.5, 'w_loc': 1, 'P1': 8, 'P2': 1, 'T': 0.35,
         'X1':0.001, 'X2': 0.5, 'Dfc': 3, 'Dcf': 1, 'Dff': 0,
         'Lfc': 1, 'Lcf': 1, 'Afc': 0, 'Acf': 0, 'Aff': 0, 'B_start': 0}
# specify what kind of weights will be used
# TODO: add documentation of what this syntax means
weights = {'fcf': {'loc': 'w_loc'}}

# Section 2.  Generating synthetic data and plotting summary
# statistics from the data

# generate synthetic recall sequences
# simulates all the study trials created in the synth_study data frame
sim = model.generate(synth_study, param, patterns=patterns, weights=weights)

# merge the study and recall events in preparation for analysis
sim_merged = fr.merge_free_recall(sim)

# serial position curve
rec_pos = fr.spc(sim_merged)
g = fr.plot_spc(rec_pos)
plt.savefig('temp_spc.pdf')

# lag-CRP curve
crp = fr.lag_crp(sim_merged)
g = fr.plot_lag_crp(crp)
g.set(ylim=(0, .6))
plt.savefig('temp_crp.pdf')


# Section 3. Using predictive simulations to perform parameter recovery
# A simple test of parameter recovery.
# create 11 model variants with different values of
# beta rec, which controls temporal reinstatement
# evaluate likelihood of the synthetic data for each model variant

# parameter sweep B_rec
temp_param = param.copy()
B_rec_vals = np.linspace(0,1,11)
logl_vals = np.zeros((len(B_rec_vals),),dtype=float)
for i in range(len(B_rec_vals)):
    temp_param['B_rec'] = B_rec_vals[i]
    logl, n = model.likelihood(sim, temp_param,
                               patterns=patterns, weights=weights)
    logl_vals[i] = logl

# demonstrate that the best-fitting value matches the generating value
#fig, ax = plt.subplot()
plt.clf()
plt.plot(B_rec_vals, logl_vals)
plt.plot([0.5, 0.5],[plt.ylim()[0], plt.ylim()[1]])
plt.xlabel('beta rec value')
plt.ylabel('log likelihood')
plt.savefig('B_rec_recovery.pdf')

# Section 4. Creating synthetic neural data and linking it to the
# temporal context reinstatement process.  Generating synthetic
# behavioral data using this 'neural' model.

# create B_rec array with as many entries as recall events

# this gets added to the param structure
# call the likelihood function, passing in the param structure (using CMRDistributed model class)
# Neal proposed we may want to use the study/recall data structure to store the param values
# this would make sense, as
# going from likelihood to likelihood_subject requires filtering the data structures
# likelihood is line 310 in fit.py
# args: data, group_param, subj_param
# data struct is filtered using the 'subject' field
# looks like prepare_sim then splits it into study events and recall events, both passed into likelihood_subject
# then in likelihood_subject (line 222 models.py) prepare_list_param is for Lfc, Lcf, p_stop
# if e.g. you wanted arbitrary variability based on a data struct field
# this should presumably happen in likelihood_subject, before call to net.study

# sim contains study events and recall events, trial_type says which is which
synth_data = sim.copy()
neural_signal_strength = 0.1
base_val = param['B_rec']
var_B_rec = base_val + np.random.randn(synth_data.shape[0]) * neural_signal_strength
var_B_rec[var_B_rec<0] = 0
var_B_rec[var_B_rec>1] = 1
synth_data = synth_data.assign(B_rec=pd.Series(var_B_rec).values)
# set B_rec column to NaN or None for study events
synth_data['B_rec'][synth_data['trial_type']=='study'] = None

# so now the goal is to have the simulation code check the data structure for
# fields that match parameter names, and use the entries to set the
# variable values of that parameter!

# B can be an array for study events, also for recall events?
# cymr/network.py

# track the sequence of function calls
# model.likelihood
# prepare_sim(subject_data)
# likelihood_subject(study, recall, param) fit.py->models.py CMRDistributed
# ... net.study()
# ... net.p_recall(~,~,param['B_rec'],...)
# p_recall() # at this point it has B, iterates over recall events
# so, either alter the call to the param structure
# or alter p_recall function to treat param as vector?

print('hi')
#param_name = ['B_enc', 'B_rec']
#param_sweep = [np.linspace(0, 1, 5), np.linspace(0, 1, 5)]

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


