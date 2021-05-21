
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cymr import network

# Tutorial code for
# Polyn, S. M. Assessing neurocognitive hypotheses in a predictive
# model of the free-recall task.

# set seed for rng

# This tutorial provides a demonstration of predictive and
# generative modeling with the Context Maintenance and Retrieval (CMR)
# model

# Section 1.  Setting model parameters and task details

# creating parameters for a basic version of CMR
# param = struct();
# B_enc: context integration rate during encoding
# B_rec: context integration rate during recall
# P1 and P2 control the primacy mechanism (a learning rate boost
# for early serial positions)
# 'sampling_rule' and T control the input to the recall competition
# 'classic' means the version described in Howard & Kahana (2002)
# (a form of softmax)
# T controls degree of non-linearity in the transformation from
# item support to probability of recall
# 'stop_rule', X1, and X2 control likelihood of recall termination
# 'op': termination probability increases steadily with output position
# Dfc and Dcf control the strength of pre-experimental associations
# (D refers to the diagonal elements of the associative matrices)
# fc / cf specifies direction of projection

# set the parameters

# param.B_enc = 0.7;
# param.B_rec = 0.5;
# param.P1 = 8;
# param.P2 = 1;
# param.sampling_rule = 'classic';
# param.T = 0.35;
# param.stop_rule = 'op';
# param.X1 = 0.001;
# param.X2 = 0.5;
# param.Dfc = 3;
# param.Dcf = 1;

model = models.CMR()
param = {'B_enc': 0.7, 'B_rec': 0.5, 'P1': 8, 'P2': 1, 'T': 0.35, 'X1':0.001, 'X2': 0.5, 'Dfc': 3, 'Dcf': 1}
# right way to specify parameters?
fixed = {'B_rec': .8, 'Afc': 0, 'Dfc': 1, 'Acf': 0, 'Dcf': 1,
         'Lfc': 1, 'Lcf': 1, 'P1': 0, 'P2': 1,
         'T': 10, 'X1': .05, 'X2': 1}

var_bounds = {'B_enc': (0, 1), 'B_rec': (0, 1), 'Dfc': (0, 10), 'Acf': (0, 10), 'Dcf': (0, 10),
              'Lfc': (0, 10), 'Lcf': (0, 10), 'P1': (0, 10), 'P2': (0, 10),
              'X1': (0, 10), 'X2': (0, 10), 'T': (0, 2)}

var_names = ['B_enc', 'B_rec', 'Dfc', 'Acf', 'Dcf', 'Lfc', 'Lcf', 'P1', 'P2', 'X1', 'X2', 'T']
#results = model.fit_indiv(mixed, fixed, var_names, var_bounds, n_jobs=4, method='de')

# have to fix this line
study_data = mixed.loc[(mixed['trial_type'] == 'study')]
recalls = model.generate(study_data, {}, subj_param)
