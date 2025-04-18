import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
#import seaborn as sns

from psifr import fr
from cymr import cmr
from cymr import parameters
from cymr import network

import tutorial_helpers as th

#from numpy.random import Generator, PCG64
#rng = Generator(PCG64(seed=42))
rng = np.random.default_rng()

# setting the path for where you want to save figures
# change this to be a folder on your computer
figpath = 'c:/Users/polynsm/modelsmemory312/Krag_tutorial/'

# SECTION 1.  Setting model parameters and task details

# param_def = parameters.Parameters()

# this helper function creates a set of synthetic patterns corresponding
# to the pool of potential study items
patterns = th.create_patterns(24)

# this helper function creates a pandas dataframe describing a set of synthetic study events,
# these will be used to guide a simulation that will generate synthetic recall sequences
n_subj = 20
n_trials = 6
list_len = 24
synth_study = th.create_expt(patterns, n_subj, n_trials, list_len)

# create the model using the CMR module
model = cmr.CMR()

# create a parameter definitions object
param_def = cmr.CMRParameters()
param_def.set_sublayers(f=['task'], c=['task'])
fc_weights = {(('task', 'item'), ('task', 'item')): 'Dfc * w_loc * loc'}
cf_weights = {(('task', 'item'), ('task', 'item')): 'Dcf * w_loc * loc'}
param_def.set_weights('fc', fc_weights)
param_def.set_weights('cf', cf_weights)
# set the model parameters to reasonable values
# B_enc: contextual integration rate for study items (enc: encoding)
# B_rec: contextual integration rate for recalled items
# w_loc: weight for the localist component of context (it is the only component of context)
# P1: primacy parameter controlling learning rate advantage for first item
# P2: primacy parameter controlling decay rate of learning rate advantage over subsequent items
# T: tau parameter controlling exponentiation of item support values during recall competition
# X1: recall termination parameter, probability that recall will terminate at first output position
# X2: recall termination parameter, controls exponential increase in recall termination prob over output positions
# Lfc: Hebbian learning rate for feature-to-context associations
# Lcf: Hebbian learning rate for context-to-feature associations 
# B_start: contextual integration rate for retrieval of start-of-list context when memory search initiates 
param_def.set_fixed(B_enc=0.7, 
                    B_rec=0.5, 
                    w_loc=1, 
                    P1=8.0, 
                    P2=1.0,
                    T=0.35, 
                    X1=0.001, 
                    X2=0.5, 
                    Lfc=1,
                    Lcf=1,
                    Dfc=3,
                    Dcf=1,
                    B_start=0)

# these parameters will get used later
neural_scaling = 0.15
orig_B_rec = param_def.fixed['B_rec']

# this weights dictionary is used to specify how to set up the weighted connections
# between units in the model.  Here, fcf is shorthand for the two weight matrices
# projecting from the feature (f) layer to the context layer (c), and back again
# i.e. fc refers to the feature-to-context projection, and cf refers to the
# context-to-feature projection.
# 'loc' means we are using localist representations for the studied items, aka
# orthonormal vectors, aka unit vectors
# param_def.weights = {'fcf': {'loc': 'w_loc'}}

# SECTION 2.  Generating synthetic data and plotting summary
# statistics from the data

# The generate function is used to generate synthetic recall sequences.
# Each row in the synth_study dataframe describes a study event (the presentation
# of a to-be-remembered word). The code iterates through these to simulate an experiment.
# The generate function returns 'sim', a dataframe containing the original study events,
# and model-generated recall events.
sim = model.generate(synth_study, param_def.fixed, param_def=param_def, patterns=patterns, n_rep=2)

# The likelihood function takes a set of study and recall events and determines
# how likely it is that the model defined by param_def (and the model code) generated
# that set of recall events.  I.e., what is the likelihood of these data given this model?
# This is a best-case scenario of sorts, in that the model being evaluated actually
# literally did generate these data.
results = model.likelihood(sim, param_def.fixed, param_def=param_def, patterns=patterns)

# This tutorial uses the psifr package to carry out basic behavioral analysis of the
# free-recall data. The dataframes produced by the models in the cymr package are
# designed to work with psifr. This merge_free_recall function does some
# pre-processing of the dataframe necessary for the psifr functions. In brief, if a
# particular study item is later recalled, the same event can contain information about
# both the original study event and the recall event (they've been merged into the
# same event)
sim_merged = fr.merge_free_recall(sim)

# calculate probability of recall as a function of serial position
# plot a serial position curve using psifr, seaborn, and matplotlib, and save it to disk
rec_pos = fr.spc(sim_merged)
g = fr.plot_spc(rec_pos)
g.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig(figpath+'spc.pdf', bbox_inches='tight')

# calculate likelihood of a recall transition of a particular lag-distance
# plot a lag-CRP curve, and save it to disk
crp = fr.lag_crp(sim_merged)
g = fr.plot_lag_crp(crp)
g.set(ylim=(0, .6))
g.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig(figpath+'crp.pdf', bbox_inches='tight')


# SECTION 3. Using predictive simulations to perform parameter recovery.
# A simple demonstration of parameter recovery.
# The code creates 11 model variants with different values of
# B_rec ("beta rec"), which controls temporal reinstatement.
# The code evaluates the likelihood of the synthetic data for each model variant

# parameter sweep B_rec
print('running parameter sweep over B_rec')
# make a copy of the parameter definitions object
param_sweep = parameters.Parameters()
param_sweep.fixed = param_def.fixed.copy()

B_rec_vals = np.linspace(0, 1, 11)
logl_vals = np.zeros((len(B_rec_vals),), dtype=float)
for i in range(len(B_rec_vals)):
    param_sweep.fixed['B_rec'] = B_rec_vals[i]
    results = model.likelihood(sim, param_sweep.fixed,
                               param_def=param_def,
                               patterns=patterns)
    logl_vals[i] = np.sum(results['logl'])
    print('*', end='')
print('')

# This figure demonstrates that the best-fitting model (largest likelihood value)
# has a B_rec parameter value that matches B_rec of the generating model
plt.clf()
plt.plot(B_rec_vals, logl_vals)
bottom, top = plt.ylim()
plt.plot([0.5, 0.5], [bottom, top], linestyle='dashed')
plt.ylim(bottom, top)
plt.xlabel(r'$\beta_{rec}$ value')
plt.ylabel('log likelihood')
plt.savefig(figpath+'B_rec_recovery.pdf', bbox_inches='tight')

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
ndf = th.create_expt(patterns, n_subj, n_trials, list_len, dummy_recalls=True)

# we create a column on the dataframe called 'hcmp' (short for hippocampus)
ndf = ndf.assign(hcmp=np.nan)
these_entries = ndf.loc[(ndf.trial_type=='recall'), 'hcmp']
# We simulate the neural signal as a stochastic process producing values
# drawn from a normal distribution with mean = 0 and stdev = 1.

#signal = np.random.randn(these_entries.shape[0])
signal = rng.standard_normal(size=these_entries.shape[0])

ndf.loc[(ndf.trial_type=='recall'), 'hcmp'] = signal

# we use param_def from above, but make some modifications
# we add a neural scaling parameter
param_def.fixed['neural_scaling'] = neural_scaling

# Then we define a dynamic parameter:

# syntax for 'dynamic' dict structure:
# {update phase: {param name: string to be evaluated}}

# The 'recall' key specifies that this dynamic parameter changes with each recall event.  
# The 'B_rec' key specifies which parameter will be updated.
# The set_dynamic function evaluates the string following 'B_rec', using the numpy
# namespace and the namespace of the parameters defined in param_def.fixed.
# The stochastic hcmp values are scaled and added to B_rec, and 'clip' 
# bounds B_rec at 0 and 1 (because the B_rec parameter is limited to this range).

param_def.set_dynamic('recall', {'B_rec': 'clip(B_rec + hcmp * neural_scaling, 0, 1)'})

# recall_keys tells the generate function to preserve the 'hcmp' column on the events dataframe
# and that the relevant values of 'hcmp' are the ones defined for recall events
# data_keys = {'recall': ['hcmp']}
recall_keys = ['hcmp']

# n_rep (number of repetitions) controls how much data is generated, e.g.
# n_rep=2 tells it to generate 2x as much data as in the dataframe provided
dyn_sim = model.generate(ndf, param_def.fixed, param_def=param_def,
                         patterns=patterns, recall_keys=recall_keys, n_rep=1)

# dyn_sim is a data structure containing study events and recall events.
# but the model code doesn't pass the 'hcmp' value back out, so
# we will have to copy it over from the ndf dataframe
dyn_sim = th.fix_hcmp_field(ndf, dyn_sim)

# demonstrate how lag-CRP is different for recall events with
# low vs high temporal reinstatement

# get the data structure ready for behavioral analysis with psifr package
# recall_keys here is used like data_keys['recall'] above (it tells psifr to
# preserve the 'hcmp' column for recall events).
dsh_merged = fr.merge_free_recall(dyn_sim, recall_keys=['hcmp'])

print('running lag-CRP analyses')
# lag-CRP curve for all recall transitions
crp = fr.lag_crp(dsh_merged)
g = fr.plot_lag_crp(crp)
g.set(ylim=(0, .6))
plt.savefig(figpath+'dsh_overall_crp.pdf', bbox_inches='tight')

# The plot_var_crp function conditionalizes the lag-CRP to filter recall transitions 
# based on the value of 'hcmp'. 
# Every transition has an item you are coming from (x) and an item you are going
# to (y). For low-temporal-reinstatement transitions there is a statement (x < -0.5) 
# which causes the analysis to only include transitions where the 'from' item has a low 
# value for hcmp / temporal reinstatement.
th.plot_var_crp(dsh_merged, figpath)

tf_all = fr.lag_rank(dsh_merged)
tf_lo = fr.lag_rank(dsh_merged, test_key='hcmp', test=lambda x, y: x < -0.5)
tf_hi = fr.lag_rank(dsh_merged, test_key='hcmp', test=lambda x, y: x > 0.5)

# SECTION 5. Running predictive simulations given the data created by the
# model with temporal reinstatement (B_rec) that varies across recall events.

# calculate likelihood under perfect case where B_rec fluctuations perfectly
# match what was used to create the synthetic data

# 'hcmp' is a recall_key, as described above
recall_keys = ['hcmp']

results = model.likelihood(dyn_sim, param_def.fixed,
                           param_def=param_def,
                           patterns=patterns,
                           recall_keys=recall_keys)

# We used a synthetic neural signal to influence a cognitive process
# in the CMR model.  Now, we imagine a scenario in which we don't have
# access to the 'true' signal, but rather only have access to a noisy
# version of the signal.

# In other words, we will distort the original neural fluctuations so
# they no longer perfectly match the fluctuations used to generate the
# data. The noisy signal is a weighted average of the original neural
# values and the noise values.

# dndf: (d)ynamic-recall-parameter with (n)oise (d)ata(f)rame
dndf = dyn_sim.copy()

# weighted mean of original value and noise value
# 0.0 would mean no contribution of noise
# 1.0 would mean no contribution of original signal
noise_weight = 0.5

hcmp_rec = dyn_sim.loc[(dyn_sim.trial_type=='recall'), 'hcmp']

#noise = np.random.randn(hcmp_rec.shape[0])
noise = rng.standard_normal(size=hcmp_rec.shape[0])

noisy_vals = (hcmp_rec * (1-noise_weight)) + (noise * noise_weight)
dndf.loc[(dndf.trial_type=='recall'), 'hcmp'] = noisy_vals

# sweep over neural scaling and B_rec base values
dnparam = param_def.copy()
# dnparam = parameters.Parameters()
# start with the same default parameters we defined in section 1
# dnparam.fixed = param_def.fixed.copy()
# dnparam.weights = param_def.weights.copy()
# dnparam.dynamic = {'recall': {'B_rec': 'clip(B_rec + hcmp * neural_scaling, 0, 1)'}}
dnparam.set_dynamic('recall', {'B_rec': 'clip(B_rec + hcmp * neural_scaling, 0, 1)'})
recall_keys = ['hcmp']

B_rec_vals = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
nscale_vals = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

logl_vals = np.zeros((len(B_rec_vals), len(nscale_vals)))

print('running parameter sweep over B_rec and neural_scaling')
for i in range(len(B_rec_vals)):
    for j in range(len(nscale_vals)):
        # set the parameters
        dnparam.fixed['B_rec'] = B_rec_vals[i]
        dnparam.fixed['neural_scaling'] = nscale_vals[j]
        # evaluate likelihood
        results = model.likelihood(dndf, dnparam.fixed,
                                   param_def=param_def,
                                   patterns=patterns,
                                   recall_keys=recall_keys)
        logl_vals[i,j] = np.sum(results['logl'])
        print('*', end='')
print('')

width = 10
precision = 7
# using python formatted strings to print results
print(f"Original model: B_rec={param_def.fixed['B_rec']}, neural_scaling={param_def.fixed['neural_scaling']}")
print(f'Best fitting neurally informed model:\n log-likelihood: {np.max(logl_vals):{width}.{precision}}')
print(f' B_rec: {B_rec_vals[np.unravel_index(np.argmax(logl_vals), logl_vals.shape)[0]]}, ', end='')
print(f'neural_scaling: {nscale_vals[np.unravel_index(np.argmax(logl_vals), logl_vals.shape)[1]]}')

# likelihood for a version of the model where B_rec is not dynamic
# i.e., the generating model had dynamic variability in B_rec, but the evaluating
# model does not, it is neurally naive

naiveparam = cmr.CMRParameters()
naiveparam.fixed = param_def.fixed.copy()
naiveparam.weights = param_def.weights.copy()
naiveparam.sublayers = param_def.sublayers.copy()
# this isn't strictly necessary as naiveparam doesn't have a dynamic param field to actually make use of neural_scaling
naiveparam.fixed['neural_scaling'] = 0

naive_logl = np.zeros((len(B_rec_vals)))

print('running parameter sweep over B_rec for neurally naive model')
for i in range(len(B_rec_vals)):
    naiveparam.fixed['B_rec'] = B_rec_vals[i]
    results = model.likelihood(dndf, naiveparam.fixed,
                               param_def=naiveparam,
                               patterns=patterns)
    naive_logl[i] = np.sum(results['logl'])
    n = np.sum(results['n'])

print(f"Original model: B_rec={param_def.fixed['B_rec']}, neural_scaling={param_def.fixed['neural_scaling']}")
print('Evaluating neurally naive model')
print(f'Best fitting model:\n log-likelihood: {np.max(naive_logl):{width}.{precision}}')
print(f' B_rec: {B_rec_vals[np.unravel_index(np.argmax(naive_logl), naive_logl.shape)[0]]}')

# SECTION 6. Model comparison
# Calculating AIC with correction for finite samples
# syntax: calc_aic(n, V, L)
# n is number of estimated data points
# V is number of free param
# L is log-likelihood

sweep_aic_vals = np.zeros((logl_vals.shape))
naive_aic_vals = np.zeros((naive_logl.shape))
for i in range(len(B_rec_vals)):
    naive_aic_vals[i] = th.calc_aic(n, 1, naive_logl[i])
    for j in range(len(nscale_vals)):
        sweep_aic_vals[i, j] = th.calc_aic(n, 2, logl_vals[i, j])

best_aic = np.zeros((2,))
# 0 is naive, 1 is neurally informed
best_aic[0] = naive_aic_vals[np.unravel_index(np.argmax(naive_logl), naive_logl.shape)[0]]
best_aic[1] = sweep_aic_vals[np.unravel_index(np.argmax(logl_vals), logl_vals.shape)]

print('AIC score for best-fitting naive model:')
print(f' AIC: {best_aic[0]:{width}.{precision}}')
print('AIC score for best-fitting neurally informed model:')
print(f' AIC: {best_aic[1]:{width}.{precision}}')

# weighted AIC for best-fit naive vs best-fit neural
temp_waic = np.exp(-0.5 * (best_aic - np.max(best_aic)))
waic = temp_waic / np.sum(temp_waic)
print(f'wAIC naive: {waic[0]:{width}.{precision}}, wAIC neural: {waic[1]:{width}.{precision}}')

# check whether n from likelihood function includes termination events

# count up the number of recall events, this is the number of data points
# we add n_trials because the model counts recall termination as a
# data point; it tries to predict this just like it tries to
# predict identity of recalled items

# SECTION 7. Scramble neural scaling values for a permutation test

# for this exploration of permutation statistics we assume we
# know the true generating parameters of B_rec and neural_scaling,
# then we can see how scrambling the neural signal affects the likelihood
# scores under otherwise perfect conditions

# set this to 100 or more for a more refined p-value
n_scrambles = 20
logl_perm = np.zeros((n_scrambles,))

orig_vals = dyn_sim.loc[(dyn_sim.trial_type=='recall'), 'hcmp'].values

dnparam.fixed['B_rec'] = orig_B_rec
dnparam.fixed['neural_scaling'] = neural_scaling

for i in range(n_scrambles):
    # shuffle around the neural signal and place it back on the dataframe
    #shuffle_inds = np.random.permutation(orig_vals.shape[0])
    shuffle_inds = rng.permutation(orig_vals.shape[0])
    shuf_vals = orig_vals[shuffle_inds]
    dndf.loc[(dndf.trial_type == 'recall'), 'hcmp'] = shuf_vals
    # calculate likelihood of data given model with shuffled neural signal
    results = model.likelihood(dndf, dnparam.fixed,
                               param_def=dnparam,
                               patterns=patterns,
                               recall_keys=recall_keys)
    logl_perm[i] = np.sum(results['logl'])
    print('*', end='')
print('')

# we can get a p-value out of this permutation analysis by
# comparing the log-likelihood of the model with the unscrambled
# neural signal, to the distribution of log-likelihoods with
# different scrambles of the neural signal

# The likelihoods with scrambled signal tend to be much worse than for the
# intact model, so this permutation analysis will likely produce a
# p-value of zero.  However, if you decide to explore modifications to the
# tutorial, you may find cases where this final analysis is more informative

best_logl = np.max(logl_vals)
# count up the number of times the scrambled logl value exceeds the
# original logl value
pval = np.sum(best_logl < logl_perm) / n_scrambles

print(f'Best log-likelihood with scrambled neural signal: {np.max(logl_perm):{width}.{precision}}')
print(f'p-value for neurally informed model against permutation distribution: {pval:{width}.{precision}}')

print('A good place for a final breakpoint.')


