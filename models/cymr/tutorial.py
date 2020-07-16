import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from psifr import fr
from cymr import cmr
from cymr import parameters
from cymr import network

import synth_data_convenience as sdc

# Section 1.  Setting model parameters and task details

# This works!
# tester_fixed = {'B_rec': 0.5, 'PX': 8, 'neural_scaling': 0.1}
# tester_rec = {'input': np.array([0, 1, 2], dtype=int)}
# gen_dynamic = {'recall': {'B_rec': 'clip(B_rec + random.randn() * neural_scaling, 0, 1)'}}
# tester_dynam = parameters.set_dynamic(tester_fixed, tester_rec, gen_dynamic['recall'])

param_def = parameters.Parameters()
# param_def.add_fixed(B_enc=0.5)
# param_def.fixed['B_enc']

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
model = cmr.CMRDistributed()

# define the parameters
# TODO: add documentation explaining what the parameters do
param_def.fixed = {'B_enc': 0.7, 'B_rec': 0.5,
                   'w_loc': 1, 'P1': 8, 'P2': 1,
                   'T': 0.35, 'X1': 0.001, 'X2': 0.5,
                   'Dfc': 3, 'Dcf': 1, 'Dff': 0,
                   'Lfc': 1, 'Lcf': 1, 'Afc': 0,
                   'Acf': 0, 'Aff': 0, 'B_start': 0}
# specify what kind of weights will be used
# TODO: add documentation of what this syntax means
param_def.weights = {'fcf': {'loc': 'w_loc'}}

# Section 2.  Generating synthetic data and plotting summary
# statistics from the data

# generate synthetic recall sequences
# simulates all the study trials created in the synth_study data frame
# sim is a dataframe containing both study and recall events
sim = model.generate(synth_study, param_def, patterns=patterns, weights=param_def.weights)

# calculate the likelihood of the model defined by param_def (and model code)
# given the data in sim (produced by this very same model)
logl, n = model.likelihood(sim, param_def, patterns=patterns, weights=param_def.weights)

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
param_sweep = parameters.Parameters()
param_sweep.fixed = param_def.fixed.copy()
param_sweep.weights = param_def.weights.copy()
B_rec_vals = np.linspace(0, 1, 11)
logl_vals = np.zeros((len(B_rec_vals),), dtype=float)
for i in range(len(B_rec_vals)):
    param_sweep.fixed['B_rec'] = B_rec_vals[i]
    logl, n = model.likelihood(sim, param_sweep,
                               patterns=patterns, weights=param_sweep.weights)
    logl_vals[i] = logl

# demonstrate that the best-fitting value matches the generating value
# fig, ax = plt.subplot()
plt.clf()
plt.plot(B_rec_vals, logl_vals)
plt.plot([0.5, 0.5], [plt.ylim()[0], plt.ylim()[1]])
plt.xlabel('beta rec value')
plt.ylabel('log likelihood')
plt.savefig('B_rec_recovery.pdf')

# Section 4. Creating synthetic neural data and linking it to the
# temporal context reinstatement process.  Generating synthetic
# behavioral data using this 'neural' model.

# sim contains study events and recall events created by our generative simulation
# trial_type says which is which
synth_study2 = synth_study.copy()
# strip off the recall events
# synth_data = synth_data.loc[synth_data['trial_type']=='study']
# we will use param_def from above, but make some modifications
param_def.fixed['neural_scaling'] = 0.1

param_def.dynamic = {'recall': {'B_rec': 'clip(B_rec + hcmp * neural_scaling, 0, 1)'}}

# make the synthetic neural values for the hcmp data column
# it will get added to baseline val, scaled, and clipped within the code
var_signal = np.random.randn(synth_study2.shape[0])
synth_study2 = synth_study2.assign(hcmp=pd.Series(var_signal).values)
# now the B_rec parameter will vary from recall event to recall event
# controlled by the values in the 'hcmp' column of the data structure

# syntax for 'dynamic' dict structure:
# {update phase: {param name: [data column, optional args]}}

# if you want a dynamic recall param, but do not need to recover the sequence of random numbers:
# gen_dynamic = {'recall': {'B_rec': 'clip(B_rec + random.randn() * neural_scaling, 0, 1)'}}
# this tells the code that the synthetic neural signal (the 'hcmp' column on the data structure)
# is attached to the study events. Generative simulations create recall events, so they don't
# take recall events as inputs. For this version of the model we aren't allowing errors,
# so the max number of recall events is the same as the number of study events.
# As such, if you have a generative simulation with a dynamic parameter that changes
# from recall event to recall event, and requires externally provided values,
# the dynamic parameter evaluation code will check the study structure for those
# values. It will use 'position' on the study events to reference output position of
# the recall events.
data_keys = {'study': ['hcmp'], 'recall': []}
dyn_sim = model.generate(synth_study2, param_def, patterns=patterns,
                         weights=param_def.weights, data_keys=data_keys)

# dyn_sim is a data structure containing study events and recall events.
# As described above, the dynamic 'B_rec' parameter values fluctuate during recall
# but the synthetic 'hcmp' neural signal was specified as a column on
# the study events.  The custom code below iterates through the recall
# events, finds the corresponding study event, and copies over the 'hcmp' value.

# dsh: Dynamic-parameter Simulation with synthetic Hippocampal activity
dsh = dyn_sim.copy()
# iterate over dyn_sim, alter dsh
for index, row in dyn_sim.iterrows():
    # print('hi')
    if row['trial_type']=='recall':
        # filter to get the study event with this
        # subject, list, position
        m1 = dyn_sim['subject']==row['subject']
        m2 = dyn_sim['list']==row['list']
        m3 = dyn_sim['trial_type']=='study'
        m4 = dyn_sim['position']==row['position']
        mask = m1 & m2 & m3 & m4
        # mask should return a single value
        val = dyn_sim['hcmp'][mask]
        dsh.loc[index,('hcmp')] = val.to_numpy()[0]

# dsh now is a data structure with synthetic neural values on the recall events
# demonstrate how lag-CRP is different for recall events with low vs high temporal
# reinstatement

# get the data structure ready for behavioral analysis with psifr package
dsh_merged = fr.merge_free_recall(dsh, recall_keys=['hcmp'])

# lag-CRP curve for all recall transitions
crp = fr.lag_crp(dsh_merged)
g = fr.plot_lag_crp(crp)
g.set(ylim=(0, .6))
plt.savefig('dsh_overall_crp.pdf')

# here we tell the lag_crp function that we want it to keep track of the
# hcmp field on the data, and keep this associated with the items being
# transitioned between.  Then we define a test function using the lambda
# keyword.  Here, x and y correspond to the identities of the 2 items being
# transitioned between.  Every transition has an item you are coming from (x)
# and an item you are going to (y).  Here we only need to test the hcmp value
# associated with x, so we are only including transitions where the 'from' item
# has a low value of temporal reinstatement.
lo_crp = fr.lag_crp(dsh_merged, test_key='hcmp', test=lambda x, y: x < -0.5)
g = fr.plot_lag_crp(lo_crp)
g.set(ylim=(0, .6))
plt.savefig('dsh_low_hcmp_crp.pdf')

# this just changes the test to only include transitions where the 'from' item
# has a high value of temporal reinstatement.
hi_crp = fr.lag_crp(dsh_merged, test_key='hcmp', test=lambda x, y: x > 0.5)
g = fr.plot_lag_crp(hi_crp)
g.set(ylim=(0, .6))
plt.savefig('dsh_high_hcmp_crp.pdf')

print('hi')

# Section 5. Predictive simulations given the data from the model
# with variable temporal reinstatement.

# calculate likelihood under perfect case where B_rec fluctuations perfectly
# match what was used to create synthetic data

# calculate the likelihood
# 'hcmp' becomes a recall_key
data_keys = {'study': [], 'recall': ['hcmp']}
print('hi')
logl, n = model.likelihood(dsh, param_def, patterns=patterns,
                           weights=param_def.weights, data_keys=data_keys)

# distort the original neural fluctuations so they no longer match

# try different values of neural_scaling parameter to see which gives best fit

# sweep over neural scaling and B_rec base values

# likelihood for a version of the model where B_rec is not dynamic
# neurally naive

# Section 6.
# AIC with correction for finite samples
# n is number of estimated data points
# V is number of free param
# L is log-likelihood
# 2*L + 2*V + (2*V*(V+1)) / (n-V-1)

# count up the number of recall events, this is the number of data points
# we add n_trials because the model counts recall termination as a
# data point; it tries to predict this just like it tries to
# predict identity of recalled items

# Section 7. Scramble neural scaling values for a permutation test

print('hi')



##
# comments to sift through later
##

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


