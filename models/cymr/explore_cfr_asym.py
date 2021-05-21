
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from psifr import fr

from cfr import task
from cfr import figures
from cfr import framework

sim_dir = '/Users/polyn/Science/Analysis/catFR/dropbox/v3/cmr_fcf-loc-cat-use/'
fig_dir = '/Users/polyn/Science/Analysis/catFR/dropbox/v3/cmr_fcf-loc-cat-use/'
data_dir = '/Users/polyn/Science/Analysis/catFR/dropbox/'
data_fig_dir = '/Users/polyn/Science/Analysis/catFR/figs/'
# sim_file = 'sim.csv'
data_file = 'cfr_eeg_mixed.csv'

# print('read simulated free recall data')
sim = task.read_free_recall(sim_dir+'sim.csv')

# load trials to simulate
data = pd.read_csv(data_dir+data_file)
study_data = data.loc[(data['trial_type'] == 'study')]

# get model, patterns, and weights
model = cmr.CMRDistributed()
patterns = network.load_patterns(patterns_file)
param_file = os.path.join(fit_dir, 'parameters.json')
param_def = parameters.read_json(param_file)

# load the best-fit parameters (fit.csv)


