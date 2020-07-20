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

# SECTION 1.  Setting model parameters and task details

param_def = parameters.Parameters()

# this helper function creates a set of synthetic patterns corresponding
# to the pool of potential study items
patterns = sdc.create_patterns(24)

# this helper function creates a pandas dataframe describing a set of synthetic study events,
# these will be used to guide a simulation that will generate synthetic recall sequences
n_subj = 20
n_trials = 6
list_len = 24
synth_study = sdc.create_expt(patterns, n_subj, n_trials, list_len)

# create the model using the CMR-Distributed module
model = cmr.CMRDistributed()

# create a parameter definitions object
param_def = parameters.Parameters()
# set the model parameters to reasonable values
# B_enc: ...
# B_rec: ...
# TODO: add documentation explaining what the parameters do
param_def.fixed = {'B_enc': 0.7, 'B_rec': 0.5,
                   'w_loc': 1, 'P1': 8, 'P2': 1,
                   'T': 0.35, 'X1': 0.001, 'X2': 0.5,
                   'Dfc': 3, 'Dcf': 1, 'Dff': 0,
                   'Lfc': 1, 'Lcf': 1, 'Afc': 0,
                   'Acf': 0, 'Aff': 0, 'B_start': 0}

# this weights dictionary is used to specify how to set up the weighted connections
# between units in the model.  Here, fcf is shorthand for the two weight matrices
# projecting from the feature (f) layer to the context layer (c), and back again
# i.e. fc refers to the feature-to-context projection, and cf refers to the
# context-to-feature projection.
# 'loc' means we are using localist representations for the studied items, aka
# orthonormal vectors, aka unit vectors
param_def.weights = {'fcf': {'loc': 'w_loc'}}

# SECTION 2.  Generating synthetic data and plotting summary
# statistics from the data

# The generate function is used to generate synthetic recall sequences.
# Each row in the synth_study dataframe describes a study event (the presentation
# of a to-be-remembered word). The code iterates through these to simulate an experiment.
# The generate function returns 'sim', a dataframe containing the original study events,
# and model-generated recall events.
sim = model.generate(synth_study, param_def, patterns=patterns, weights=param_def.weights)

# The likelihood function takes a set of study and recall events and determines
# how likely it is that the model defined by param_def (and the model code) generated
# that set of recall events.  I.e., what is the likelihood of these data given this model?
# This is a best-case scenario of sorts, in that the model being evaluated actually
# literally did generate these data.
logl, n = model.likelihood(sim, param_def, patterns=patterns, weights=param_def.weights)

# This tutorial uses the psifr package to carry out basic behavioral analysis of the
# free-recall data. The dataframes produced by the models in the cymr package are
# designed to work with psifr. This merge_free_recall function does some
# pre-processing of the dataframe necessary for the psifr functions. In brief, if a
# particular study item is later recalled, the same event can contain information about
# both the original study event and the recall event (they've been merged into the
# same event)
sim_merged = fr.merge_free_recall(sim)

# calculate probability of recall as a function of serial position
# plot a serial position curve, and save it to disk
rec_pos = fr.spc(sim_merged)
g = fr.plot_spc(rec_pos)
plt.savefig('temp_spc.pdf')

# calculate likelihood of a recall transition of a particular lag-distance
# plot a lag-CRP curve, and save it to disk
crp = fr.lag_crp(sim_merged)
g = fr.plot_lag_crp(crp)
g.set(ylim=(0, .6))
plt.savefig('temp_crp.pdf')


# SECTION 3. Using predictive simulations to perform parameter recovery.
# A simple demonstration of parameter recovery.
# The code creates 11 model variants with different values of
# B_rec ("beta rec"), which controls temporal reinstatement.
# The code evaluates the likelihood of the synthetic data for each model variant

# parameter sweep B_rec
# make a copy of the parameter definitions object
param_sweep = parameters.Parameters()
param_sweep.fixed = param_def.fixed.copy()
param_sweep.weights = param_def.weights.copy()
B_rec_vals = np.linspace(0, 1, 11)
logl_vals = np.zeros((len(B_rec_vals),), dtype=float)
for i in range(len(B_rec_vals)):
    param_sweep.fixed['B_rec'] = B_rec_vals[i]
    logl, n = model.likelihood(sim, param_sweep,
                               patterns=patterns,
                               weights=param_sweep.weights)
    logl_vals[i] = logl

# This figure demonstrates that the best-fitting model (largest likelihood value)
# has a B_rec parameter value that matches B_rec of the generating model
plt.clf()
plt.plot(B_rec_vals, logl_vals)
plt.plot([0.5, 0.5], [plt.ylim()[0], plt.ylim()[1]])
plt.xlabel('beta rec value')
plt.ylabel('log likelihood')
plt.savefig('B_rec_recovery.pdf')

# SECTION 4. Creating synthetic neural data and creating a neural linking parameter
# that allows the neural data to control the model's temporal context reinstatement
# process.  First we generate synthetic behavioral data using this 'neurally informed'
# model.

# The helper function has an option to create a set of dummy recall events.
# This model can only produce as many recall events as there were study events (as it
# is a simplified model that does not make repeats or intrusion errors).  We will create
# synthetic neural signal for each dummy recall event, then our generative simulation will
# use these synthetic neural signals during recall-sequence generation.
# ndf = 'neural data frame'
ndf = sdc.create_expt(patterns, n_subj, n_trials, list_len, dummy_recalls=True)

# We simulate the neural signal as a stochastic process producing values
# drawn from a normal distribution with mean = 0 and stdev = 1.
var_signal = np.random.randn(ndf.shape[0])
# then we create a column on the dataframe called 'hcmp' (short
# for hippocampus)
ndf = ndf.assign(hcmp=pd.Series(var_signal).values)
# for these simulations we only want to keep the values associated with
# the recall events, can set the hcmp values for study events to be 'missing'
ndf.loc[ndf['trial_type']=='study', 'hcmp'] = np.nan

# we will use param_def from above, but make some modifications
# we add a neural scaling parameter
param_def.fixed['neural_scaling'] = 0.2

# then we define a dynamic parameter

# the 'recall' key specifies that this dynamic parameter changes with each
# recall event.  The 'B_rec' key specifies which parameter will be updated.
# The set_dynamic_param code will evaluate the string, using the numpy
# namespace and the namespace of the parameters defined in param_def.fixed.
# The stochastic hcmp values are scaled and added to B_rec, and the resultant
# value is bounded at 0 and 1 (because the B_rec parameter is limited to this
# range).

# syntax for 'dynamic' dict structure:
# {update phase: {param name: string to be evaluated}}

param_def.dynamic = {'recall': {'B_rec': 'clip(B_rec + hcmp * neural_scaling, 0, 1)'}}

# now the B_rec parameter will vary from recall event to recall event
# controlled by the values in the 'hcmp' column of the data structure

# this tells the generate function to preserve the 'hcmp' column on the events dataframe
# and that the relevant values are the ones defined for recall events
data_keys = {'recall': ['hcmp']}

# n_rep controls how much data is generated, e.g.
# n_rep=2 tells it to generate 2x as much data as in the dataframe provided
dyn_sim = model.generate(ndf, param_def, patterns=patterns,
                         weights=param_def.weights, data_keys=data_keys,
                         n_rep=1)

# dyn_sim is a data structure containing study events and recall events.
# but the model code doesn't know to report the 'hcmp' value back out, so
# we will have to copy it over from the ndf dataframe

# to identify a recall event, subject, list, trial_type, position
for index, row in ndf.iterrows():
    if row.trial_type == 'recall':
        # find the corresponding row or rows in dyn_sim (rows if n_rep > 1)
        mask = (dyn_sim.subject==row.subject) & \
               (dyn_sim.list==row.list) & \
               (dyn_sim.position==row.position)
        dyn_sim.loc[mask, 'hcmp'] = row.hcmp

# dyn_sim.loc[mask, 'hcmp'] = this_val
# demonstrate how lag-CRP is different for recall events with low vs high temporal
# reinstatement

# get the data structure ready for behavioral analysis with psifr package
# recall_keys here is used like data_keys['recall'] above (it tells psifr to
# preserve the 'hcmp' column for recall events).
dsh_merged = fr.merge_free_recall(dyn_sim, recall_keys=['hcmp'])

# lag-CRP curve for all recall transitions
crp = fr.lag_crp(dsh_merged)
g = fr.plot_lag_crp(crp)
g.set(ylim=(0, .6))
plt.savefig('dsh_overall_crp.pdf')

# Here we conditionalize the lag-CRP to filter recall transitions based on the
# value of 'hcmp'. We define a test function using the lambda keyword.  Here,
# x and y correspond to the identities of the 2 items being transitioned between.
# Every transition has an item you are coming from (x) and an item you are going
# to (y). The (x < -0.5) statement will cause the analysis to only include
# transitions where the 'from' item has a low value for temporal reinstatement.
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


# SECTION 5. Running predictive simulations given the data created by the
# model with temporal reinstatement (B_rec) that varies across recall events.

# calculate likelihood under perfect case where B_rec fluctuations perfectly
# match what was used to create the synthetic data

# 'hcmp' is a recall_key
data_keys = {'study': [], 'recall': ['hcmp']}

logl, n = model.likelihood(dyn_sim, param_def, patterns=patterns,
                           weights=param_def.weights, data_keys=data_keys)

# distort the original neural fluctuations so they no longer match
# this is literally adding random normal deviates to the signal used
# to create the original behavioral data
# original signal is in the var_signal variable
scale_distortion = 0.1
distortion = np.random.randn(var_signal.shape[0]) * scale_distortion

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


