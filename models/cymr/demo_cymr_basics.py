
import numpy as np
import matplotlib.pyplot as plt

from psifr import fr
from cymr import cmr
from cymr import parameters
from cymr import network

import synth_data_convenience as sdc

param_def = parameters.Parameters()

patterns = sdc.create_patterns(30)

n_subj = 20
n_trials = 15
list_len = 30
synth_study = sdc.create_expt(patterns, n_subj, n_trials, list_len)

model = cmr.CMRDistributed()

param_def = parameters.Parameters()
param_def.fixed = {'B_enc': 0.7, 'B_rec': 0.5,
                   'w_loc': 1, 'P1': 8, 'P2': 1,
                   'T': 0.35, 'X1': 0.001, 'X2': 0.5,
                   'Dfc': 3, 'Dcf': 1, 'Dff': 0,
                   'Lfc': 1, 'Lcf': 1, 'Afc': 0,
                   'Acf': 0, 'Aff': 0, 'B_start': 0}

param_def.weights = {'fcf': {'loc': 'w_loc'}}

sim_data = model.generate(synth_study, param_def, patterns=patterns, weights=param_def.weights)

sim_merged = fr.merge_free_recall(sim_data)

rec_pos = fr.spc(sim_merged)
#g = fr.plot_spc(rec_pos)
#plt.savefig('temp_spc.pdf')


param_def.add_free(B_enc=(0, 1))


#                  'P1': (0, 15), 'P2': (0, 3),
#                  'T': (0.01, 0.5)}
print('running fit 1: \nB_enc (0, 1)')
results1 = model.fit_indiv(sim_data, param_def,
                           patterns=patterns, n_jobs=2)

# histogram of best-fit values of B_enc
fig, ax = plt.subplots()
num_bins = 10
n, bins, patches = ax.hist(results1['B_enc'].values, num_bins)
# ax.plot(bins)

ax.set_xlabel(r'$\beta_{enc}$')
ax.set_ylabel('Frequency')
ax.set_title('Degree of parameter scatter')

ax.axvline(x=param_def.fixed['B_enc'], color='b', linestyle='dashed', linewidth=2)
# fig.show()
fig.savefig('B_enc_hist_fit1.pdf')
# print('hi')

param_def.add_free(B_rec=(0, 1))
print('running fit 2: \nB_enc (0, 1)\nB_rec (0, 1)')
results2 = model.fit_indiv(sim_data, param_def,
                           patterns=patterns, n_jobs=2)

# scatterplot fitted values of B_enc and B_rec
ax.cla()
ax.scatter(results2['B_enc'], results2['B_rec'])
ax.plot(param_def.fixed['B_enc'], param_def.fixed['B_rec'],
        'ro')
fig.savefig('B_scat_fit2.pdf')

param_def.add_free(P1=(0, 15), P2=(0, 3))
print('running fit 3: \nP1 (0, 15)\nP2 (0, 3)')
results3 = model.fit_indiv(sim_data, param_def,
                           patterns=patterns, n_jobs=2)


print('hi')
